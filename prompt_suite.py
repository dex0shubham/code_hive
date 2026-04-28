"""
Code Hivemind — Prompt Suite
=============================
30 open-ended coding prompts across 6 categories.

KEY DESIGN PRINCIPLE: Each prompt must admit MANY valid solutions.
Unlike HumanEval (single correct answer), these tasks have unbounded
solution spaces — different algorithms, architectures, naming conventions,
and stylistic choices are all equally valid.

We follow the Infinity-Chat methodology: real-world-style queries
that a developer might actually ask an LLM.
"""

from dataclasses import dataclass

@dataclass
class CodePrompt:
    id: str
    category: str
    prompt: str
    language: str          # target language ("python", "javascript", "any")
    expected_diversity: str  # "high", "medium" — our hypothesis
    notes: str             # what we expect to vary


PROMPTS: list[CodePrompt] = [

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 1: OPEN_DESIGN — "Build X" with many valid designs
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="OD-01",
        category="OPEN_DESIGN",
        prompt="Write a Python class that implements an in-memory cache with expiration. Include methods for get, set, and cleanup of expired entries.",
        language="python",
        expected_diversity="high",
        notes="LRU vs TTL vs LFU, threading, data structures (dict, OrderedDict, heap), cleanup strategy (lazy vs eager)"
    ),
    CodePrompt(
        id="OD-02",
        category="OPEN_DESIGN",
        prompt="Build a simple task queue in Python that supports adding tasks with priorities, executing them in order, and retrying failed tasks up to 3 times.",
        language="python",
        expected_diversity="high",
        notes="heapq vs sorted list vs deque, retry logic (decorator vs inline), error handling patterns"
    ),
    CodePrompt(
        id="OD-03",
        category="OPEN_DESIGN",
        prompt="Create a Python rate limiter that supports both fixed-window and sliding-window strategies. It should be usable as a decorator.",
        language="python",
        expected_diversity="high",
        notes="token bucket vs leaky bucket vs fixed window, threading primitives, decorator patterns"
    ),
    CodePrompt(
        id="OD-04",
        category="OPEN_DESIGN",
        prompt="Write a JavaScript module that implements a pub/sub event system with support for wildcard subscriptions and once-only listeners.",
        language="javascript",
        expected_diversity="high",
        notes="Map vs object, regex vs trie for wildcards, memory management patterns"
    ),
    CodePrompt(
        id="OD-05",
        category="OPEN_DESIGN",
        prompt="Implement a simple key-value store in Python that persists data to disk. It should handle concurrent reads and writes safely.",
        language="python",
        expected_diversity="high",
        notes="JSON vs pickle vs sqlite vs append-only log, locking strategies, file formats"
    ),

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 2: ALGORITHM — Multiple valid algorithmic approaches
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="AL-01",
        category="ALGORITHM",
        prompt="Write a Python function that finds all anagram groups in a list of words. Return groups sorted by size (largest first).",
        language="python",
        expected_diversity="medium",
        notes="sorted-key vs char-count vs prime-product hashing, groupby vs defaultdict"
    ),
    CodePrompt(
        id="AL-02",
        category="ALGORITHM",
        prompt="Implement a function that finds the k most frequent elements in a stream of integers. Optimize for memory when the stream is very large.",
        language="python",
        expected_diversity="high",
        notes="Counter vs heap vs Count-Min Sketch vs Space-Saving, streaming vs batch"
    ),
    CodePrompt(
        id="AL-03",
        category="ALGORITHM",
        prompt="Write a Python function that detects cycles in a directed graph. The graph is given as an adjacency list (dictionary).",
        language="python",
        expected_diversity="medium",
        notes="DFS coloring vs topological sort vs iterative DFS, recursion vs stack"
    ),
    CodePrompt(
        id="AL-04",
        category="ALGORITHM",
        prompt="Implement a text similarity function that compares two documents and returns a score between 0 and 1. Choose whatever approach you think is best.",
        language="python",
        expected_diversity="high",
        notes="cosine TF-IDF vs Jaccard vs edit distance vs embedding-based vs BM25"
    ),
    CodePrompt(
        id="AL-05",
        category="ALGORITHM",
        prompt="Write a function that generates all valid combinations of n pairs of balanced parentheses. Return them as a list of strings.",
        language="python",
        expected_diversity="medium",
        notes="recursive backtracking vs iterative, naming choices, early termination"
    ),

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 3: REFACTOR — Improve given code (many valid improvements)
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="RF-01",
        category="REFACTOR",
        prompt="""Refactor this Python code to be cleaner, more Pythonic, and more maintainable:

```python
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i]['status'] == 'active':
            if data[i]['score'] > 50:
                temp = {}
                temp['name'] = data[i]['name'].upper()
                temp['score'] = data[i]['score'] * 1.1
                temp['grade'] = 'A' if data[i]['score'] > 90 else 'B' if data[i]['score'] > 70 else 'C'
                result.append(temp)
    result.sort(key=lambda x: x['score'], reverse=True)
    return result
```""",
        language="python",
        expected_diversity="high",
        notes="list comp vs filter/map, dataclass vs dict vs namedtuple, naming choices, decomposition level"
    ),
    CodePrompt(
        id="RF-02",
        category="REFACTOR",
        prompt="""Refactor this JavaScript code to use modern patterns and improve readability:

```javascript
function fetchUserData(userId, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/users/' + userId);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                var posts = [];
                for (var i = 0; i < data.postIds.length; i++) {
                    var xhr2 = new XMLHttpRequest();
                    xhr2.open('GET', '/api/posts/' + data.postIds[i]);
                    xhr2.onreadystatechange = function(postId) {
                        return function() {
                            if (this.readyState === 4 && this.status === 200) {
                                posts.push(JSON.parse(this.responseText));
                                if (posts.length === data.postIds.length) {
                                    callback(null, {user: data, posts: posts});
                                }
                            }
                        };
                    }(data.postIds[i]);
                    xhr2.send();
                }
            } else {
                callback(new Error('Failed: ' + xhr.status));
            }
        }
    };
    xhr.send();
}
```""",
        language="javascript",
        expected_diversity="high",
        notes="async/await vs Promise.all, fetch vs axios, error handling patterns, destructuring"
    ),
    CodePrompt(
        id="RF-03",
        category="REFACTOR",
        prompt="""Improve this Python code. Focus on whatever you think matters most:

```python
import re
def validate_and_parse_email_list(email_string):
    emails = email_string.split(',')
    valid = []
    invalid = []
    for e in emails:
        e = e.strip()
        if e != '':
            if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$', e):
                parts = e.split('@')
                domain = parts[1]
                local = parts[0]
                valid.append({'email': e, 'local': local, 'domain': domain})
            else:
                invalid.append(e)
    return {'valid': valid, 'invalid': invalid, 'total': len(valid) + len(invalid)}
```""",
        language="python",
        expected_diversity="high",
        notes="dataclass vs dict, email-validator lib vs regex, generator vs list, type hints"
    ),
    CodePrompt(
        id="RF-04",
        category="REFACTOR",
        prompt="Take this function and make it production-ready. Add whatever you think is needed:\n\n```python\ndef retry(func, times=3):\n    for i in range(times):\n        try:\n            return func()\n        except:\n            if i == times - 1:\n                raise\n```",
        language="python",
        expected_diversity="high",
        notes="decorator pattern, backoff strategies, logging, type hints, exception filtering"
    ),
    CodePrompt(
        id="RF-05",
        category="REFACTOR",
        prompt="""Rewrite this to be more maintainable and testable:

```python
def calculate_shipping(weight, destination, is_prime, is_fragile):
    if destination == 'domestic':
        if weight < 1:
            cost = 5.99
        elif weight < 5:
            cost = 8.99
        elif weight < 20:
            cost = 12.99
        else:
            cost = 12.99 + (weight - 20) * 0.50
    elif destination == 'international':
        if weight < 1:
            cost = 15.99
        elif weight < 5:
            cost = 25.99
        elif weight < 20:
            cost = 45.99
        else:
            cost = 45.99 + (weight - 20) * 2.00
    if is_fragile:
        cost = cost * 1.5
    if is_prime:
        cost = cost * 0.8
    return round(cost, 2)
```""",
        language="python",
        expected_diversity="high",
        notes="strategy pattern vs lookup table vs config-driven, class vs functions, enum usage"
    ),

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 4: NAMING — Tasks where naming/style is unconstrained
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="NM-01",
        category="NAMING",
        prompt="Write a Python class that represents a playing card and a deck of cards. Include methods for shuffling, drawing, and checking if the deck is empty.",
        language="python",
        expected_diversity="high",
        notes="class names, method names, enum vs string for suits/ranks, __repr__ vs __str__"
    ),
    CodePrompt(
        id="NM-02",
        category="NAMING",
        prompt="Create a Python module with helper functions for working with dates: parsing various formats, calculating differences, finding business days, and formatting for display.",
        language="python",
        expected_diversity="high",
        notes="function names, parameter names, module structure, which lib to wrap"
    ),
    CodePrompt(
        id="NM-03",
        category="NAMING",
        prompt="Write a Python class that represents an HTTP response with status code, headers, body, and timing information. Include methods for checking success, parsing JSON, and pretty-printing.",
        language="python",
        expected_diversity="high",
        notes="property vs method, attribute names, dataclass vs regular class, slots"
    ),
    CodePrompt(
        id="NM-04",
        category="NAMING",
        prompt="Implement a simple logging utility in Python that supports different log levels, formatting, and output to both console and file.",
        language="python",
        expected_diversity="medium",
        notes="class names, level names, format string design, API design choices"
    ),
    CodePrompt(
        id="NM-05",
        category="NAMING",
        prompt="Write a configuration manager class in Python that loads settings from environment variables, JSON files, and defaults, with proper precedence ordering.",
        language="python",
        expected_diversity="high",
        notes="class name, method names, merge strategies, immutability patterns"
    ),

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 5: CREATIVE_TOOL — Fun/creative coding tasks
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="CT-01",
        category="CREATIVE_TOOL",
        prompt="Write a Python script that generates ASCII art from text. Make it creative and visually interesting.",
        language="python",
        expected_diversity="high",
        notes="pyfiglet vs custom, banner styles, color support, font choices"
    ),
    CodePrompt(
        id="CT-02",
        category="CREATIVE_TOOL",
        prompt="Create a Python command-line tool that generates random project names for software projects. Make it fun and memorable.",
        language="python",
        expected_diversity="high",
        notes="word lists, combination strategies, themes, CLI framework choice"
    ),
    CodePrompt(
        id="CT-03",
        category="CREATIVE_TOOL",
        prompt="Write a Python script that analyzes a Git repository and produces an interesting summary of its history — whatever you think would be cool to see.",
        language="python",
        expected_diversity="high",
        notes="which stats to show, visualization choices, git lib vs subprocess"
    ),
    CodePrompt(
        id="CT-04",
        category="CREATIVE_TOOL",
        prompt="Build a small Python game that runs in the terminal. Choose any game you think would be fun to implement.",
        language="python",
        expected_diversity="high",
        notes="game choice itself is a diversity signal — snake, hangman, tic-tac-toe, etc."
    ),
    CodePrompt(
        id="CT-05",
        category="CREATIVE_TOOL",
        prompt="Write a Python script that takes a text file and creates an interesting visualization or analysis of its contents. Be creative.",
        language="python",
        expected_diversity="high",
        notes="word clouds, frequency analysis, readability stats, sentiment, network graphs"
    ),

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY 6: SYSTEM_DESIGN — High-level design decompositions
    # ═══════════════════════════════════════════════════════════════

    CodePrompt(
        id="SD-01",
        category="SYSTEM_DESIGN",
        prompt="Design and implement a simple URL shortener in Python. Include the shortening logic, storage, and a basic HTTP server to handle redirects.",
        language="python",
        expected_diversity="high",
        notes="hash function choice, storage (dict vs sqlite vs file), framework (flask vs http.server vs fastapi)"
    ),
    CodePrompt(
        id="SD-02",
        category="SYSTEM_DESIGN",
        prompt="Build a simple static site generator in Python that converts Markdown files to HTML pages with a template system.",
        language="python",
        expected_diversity="high",
        notes="template engine choice, markdown lib, file watching, CSS approach"
    ),
    CodePrompt(
        id="SD-03",
        category="SYSTEM_DESIGN",
        prompt="Implement a simple REST API framework in Python from scratch (without Flask/FastAPI). Support routing, JSON parsing, and basic middleware.",
        language="python",
        expected_diversity="high",
        notes="routing (regex vs trie vs dict), decorator style, middleware pattern, error handling"
    ),
    CodePrompt(
        id="SD-04",
        category="SYSTEM_DESIGN",
        prompt="Create a simple database migration tool in Python that tracks and applies schema changes using numbered SQL files.",
        language="python",
        expected_diversity="high",
        notes="state tracking (table vs file), rollback support, CLI design, error recovery"
    ),
    CodePrompt(
        id="SD-05",
        category="SYSTEM_DESIGN",
        prompt="Build a basic web scraper framework in Python that handles pagination, rate limiting, and saves results to CSV. Design it to be reusable for different sites.",
        language="python",
        expected_diversity="high",
        notes="requests vs httpx vs aiohttp, BS4 vs lxml, class hierarchy, config approach"
    ),
]


# ── System prompt ──────────────────────────────────────────────────
# Kept minimal to avoid constraining style/approach.

SYSTEM_PROMPT = (
    "You are a software developer. Write clean, working code. "
    "Choose whatever approach, libraries, patterns, and naming "
    "conventions you think are best. Only output the code."
)


def get_prompts_by_category(category: str) -> list[CodePrompt]:
    return [p for p in PROMPTS if p.category == category]


def get_all_prompt_ids() -> list[str]:
    return [p.id for p in PROMPTS]