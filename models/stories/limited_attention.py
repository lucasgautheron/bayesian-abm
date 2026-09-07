"""Weng-style meme competition under finite screens and memories."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


N_USERS = 1_000
FOLLOWS_PER_USER = 5
RETENTION_DAYS = 1

Post = tuple[int, int]
PostQueue = deque[Post]


def preferential_attachment_followers(
    n_users: int,
    follows_per_user: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int32], ...]:
    """Generate follower lists with linear preferential attachment.

    Users arrive sequentially. Each new user follows existing users sampled
    without replacement in proportion to their current follower count plus
    one. Returned entries list the followers who receive each user's posts.
    """

    followers: list[list[int]] = [[] for _ in range(n_users)]
    # One urn entry supplies each user's baseline attractiveness. Every new
    # follower adds one further entry, yielding weight followers + 1.
    attachment_urn = [0]
    for user in range(1, n_users):
        target_count = min(follows_per_user, user)
        targets: set[int] = set()
        while len(targets) < target_count:
            targets.add(
                attachment_urn[
                    int(rng.integers(0, len(attachment_urn)))
                ]
            )
        for target in targets:
            followers[target].append(user)
            attachment_urn.append(target)
        attachment_urn.append(user)

    return tuple(
        np.asarray(user_followers, dtype=np.int32)
        for user_followers in followers
    )


def _discard_expired(queue: PostQueue, step: int) -> None:
    while queue and queue[0][0] <= step:
        queue.popleft()


class LimitedAttentionModel(StoryModel):
    """Neutral meme diffusion on a network with finite user attention.

    This adapts Weng et al. (2012) to daily aggregate story counts using a
    synthetic directed preferential-attachment network because MemeTracker
    does not include users or follower links.
    """

    name = "limited_attention"
    inference_variables = ("p_new", "p_read", "p_memory")
    parameter_units = {"p_new": "probability per user-step"}

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.Uniform("p_new", lower=0.0, upper=1.0)
            pm.Uniform("p_read", lower=0.0, upper=1.0)
            pm.Uniform("p_memory", lower=0.0, upper=1.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_days = int(context["n_days"])
        p_new = parameters["p_new"]
        p_read = parameters["p_read"]
        p_memory = parameters["p_memory"]
        followers = preferential_attachment_followers(
            N_USERS,
            FOLLOWS_PER_USER,
            rng,
        )
        screens: list[PostQueue] = [deque() for _ in range(N_USERS)]
        memories: list[PostQueue] = [deque() for _ in range(N_USERS)]
        daily_mentions: list[dict[int, int]] = [
            {} for _ in range(n_days)
        ]
        retention_steps = RETENTION_DAYS * N_USERS
        story_count = 0

        def broadcast(user: int, story: int, step: int, day: int) -> None:
            expiry = step + retention_steps
            memories[user].append((expiry, story))
            for follower in followers[user]:
                screens[int(follower)].append((expiry, story))
            counts = daily_mentions[day]
            counts[story] = counts.get(story, 0) + 1

        total_steps = n_days * N_USERS
        for step in range(total_steps):
            day = step // N_USERS
            user = int(rng.integers(0, N_USERS))
            screen = screens[user]
            memory = memories[user]
            _discard_expired(screen, step)
            _discard_expired(memory, step)

            if rng.random() < p_new:
                story = story_count
                story_count += 1
                broadcast(user, story, step, day)
                continue

            visible_posts = tuple(screen)
            if not visible_posts:
                continue
            attended = rng.random(len(visible_posts)) < p_read
            for (_, visible_story), noticed in zip(
                visible_posts,
                attended,
            ):
                if not noticed:
                    continue
                if memory and rng.random() < p_memory:
                    memory_index = int(rng.integers(0, len(memory)))
                    story = memory[memory_index][1]
                else:
                    story = visible_story
                broadcast(user, story, step, day)

        mentions = np.zeros((story_count, n_days), dtype=np.float32)
        for day, counts in enumerate(daily_mentions):
            if not counts:
                continue
            stories = np.fromiter(counts, dtype=np.int64)
            mentions[stories, day] = np.fromiter(
                counts.values(),
                dtype=np.float32,
            )
        return {"mentions": mentions}


__all__ = [
    "FOLLOWS_PER_USER",
    "LimitedAttentionModel",
    "N_USERS",
    "RETENTION_DAYS",
    "preferential_attachment_followers",
]
