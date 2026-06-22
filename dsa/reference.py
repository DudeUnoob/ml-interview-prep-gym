from __future__ import annotations

from collections import Counter, deque
from heapq import heappop, heappush
from typing import Iterable


def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}
    for index, value in enumerate(nums):
        needed = target - value
        if needed in seen:
            return [seen[needed], index]
        seen[value] = index
    return []


def top_k_frequent(values: Iterable[str], k: int) -> list[str]:
    counts = Counter(values)
    heap: list[tuple[int, str]] = []
    for value, count in counts.items():
        heappush(heap, (count, value))
        if len(heap) > k:
            heappop(heap)
    return [value for _, value in sorted(heap, reverse=True)]


def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(row: int, col: int) -> None:
        if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] != "1":
            return
        grid[row][col] = "0"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dfs(row + dr, col + dc)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                islands += 1
                dfs(row, col)
    return islands


def course_schedule(num_courses: int, prerequisites: list[list[int]]) -> bool:
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, prerequisite in prerequisites:
        graph[prerequisite].append(course)
        indegree[course] += 1
    queue = deque(course for course, degree in enumerate(indegree) if degree == 0)
    taken = 0
    while queue:
        course = queue.popleft()
        taken += 1
        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)
    return taken == num_courses


def combination_sum_ii(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()
    answer: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int, total: int) -> None:
        if total == target:
            answer.append(path.copy())
            return
        for index in range(start, len(candidates)):
            if index > start and candidates[index] == candidates[index - 1]:
                continue
            value = candidates[index]
            if total + value > target:
                break
            path.append(value)
            backtrack(index + 1, total + value)
            path.pop()

    backtrack(0, 0)
    return answer
