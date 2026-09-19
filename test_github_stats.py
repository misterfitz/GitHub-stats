#!/usr/bin/python3
"""Regression tests for GitHub stats aggregation.

These tests mock the GraphQL API so they can run without ACCESS_TOKEN.
They specifically cover the concurrent-access race that caused missing
stars/repos and Python-only language stats when generate_images ran
asyncio.gather on multiple Stats property accessors.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from github_stats import Queries, Stats


def _repo(
    name: str,
    stars: int,
    forks: int,
    languages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "nameWithOwner": name,
        "stargazers": {"totalCount": stars},
        "forkCount": forks,
        "languages": {
            "edges": [
                {
                    "size": lang["size"],
                    "node": {"name": lang["name"], "color": lang.get("color", "#000")},
                }
                for lang in languages
            ]
        },
    }


PAGE_1 = {
    "data": {
        "viewer": {
            "login": "testuser",
            "name": "Test User",
            "repositories": {
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
                "nodes": [
                    _repo(
                        "testuser/python-tool",
                        stars=3,
                        forks=1,
                        languages=[{"name": "Python", "size": 1000, "color": "#3572A5"}],
                    ),
                    _repo(
                        "testuser/web-app",
                        stars=10,
                        forks=2,
                        languages=[
                            {"name": "TypeScript", "size": 5000, "color": "#2b7489"},
                            {"name": "HTML", "size": 500, "color": "#e34c26"},
                        ],
                    ),
                ],
            },
            "repositoriesContributedTo": {
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [],
            },
        }
    }
}

PAGE_2 = {
    "data": {
        "viewer": {
            "login": "testuser",
            "name": "Test User",
            "repositories": {
                "pageInfo": {"hasNextPage": False, "endCursor": "cursor2"},
                "nodes": [
                    _repo(
                        "testuser/go-service",
                        stars=7,
                        forks=0,
                        languages=[
                            {"name": "Go", "size": 3000, "color": "#00ADD8"},
                            {"name": "Python", "size": 200, "color": "#3572A5"},
                        ],
                    ),
                ],
            },
            "repositoriesContributedTo": {
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [],
            },
        }
    }
}


class DelayedQueries:
    """Mock Queries that yields between pages so concurrent accessors can race."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    async def query(self, generated_query: str) -> Dict:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if "after: \"cursor1\"" in generated_query or 'after: "cursor1"' in generated_query:
            return PAGE_2
        return PAGE_1


def _make_stats() -> Stats:
    # session is unused when queries are replaced
    stats = Stats("testuser", "fake-token", MagicMock())
    stats.queries = DelayedQueries()
    return stats


@pytest.mark.asyncio
async def test_concurrent_property_access_sees_complete_stats():
    """Mirrors generate_images.py: asyncio.gather on languages + overview fields."""
    s = _make_stats()

    async def read_overview_fields():
        return (
            await s.name,
            await s.stargazers,
            await s.forks,
            len(await s.repos),
        )

    async def read_languages():
        return await s.languages

    overview, languages = await asyncio.gather(
        read_overview_fields(), read_languages()
    )

    name, stars, forks, repo_count = overview
    assert name == "Test User"
    assert stars == 20  # 3 + 10 + 7
    assert forks == 3  # 1 + 2 + 0
    assert repo_count == 3
    assert set(languages.keys()) == {"Python", "TypeScript", "HTML", "Go"}
    assert languages["Python"]["size"] == 1200
    assert abs(sum(v["prop"] for v in languages.values()) - 100) < 0.01


@pytest.mark.asyncio
async def test_get_stats_is_idempotent_under_contention():
    s = _make_stats()
    await asyncio.gather(s.get_stats(), s.get_stats(), s.get_stats())
    assert s.queries.calls == 2  # two GraphQL pages, not duplicated
    assert await s.stargazers == 20
    assert len(await s.repos) == 3


@pytest.mark.asyncio
async def test_exclude_langs_case_insensitive():
    s = _make_stats()
    s._exclude_langs = {"html"}
    await s.get_stats()
    languages = await s.languages
    assert "HTML" not in languages
    assert "TypeScript" in languages


class _GraphQLResponse:
    async def json(self) -> Dict[str, Any]:
        return {"data": {"viewer": {"login": "testuser"}}}


class _GraphQLSession:
    def __init__(self):
        self.last_headers: Optional[Dict[str, str]] = None

    async def post(self, url: str, headers: Dict[str, str], json: Dict[str, str]):
        self.last_headers = headers
        return _GraphQLResponse()


@pytest.mark.asyncio
async def test_graphql_query_uses_bearer_token_header():
    session = _GraphQLSession()
    q = Queries("testuser", "secret-token", session)
    await q.query("{ viewer { login } }")
    assert session.last_headers == {"Authorization": "bearer secret-token"}
