## Decision model: semantic search, reranking and classification over captions

A classifier model judges each text you send against a criterion you write in natural language and returns a probability (or a choice or a score). Over the minutes of a range it works as a semantic search or a reranker: describe what you are looking for, score every minute of the range, and print only the best-scoring minutes with their ids. It also works as a filter: keep the minutes whose probability passes a threshold. It does not see the footage or the question, only the text you send, and each text is judged on its own.

A range is any part of the index you choose: a span of hours, one whole video or day when the index has several, or the minutes a broad keyword filter kept. Send one range at a time, never the whole index. A range of N minutes sends N requests, 16 at a time.

The library `/decision/decide.py` does the requests (no key is needed). Use it from a Python exec:

```python
"""Search one video/day for minutes that match a description."""
import json, sys; sys.path.insert(0, '/decision'); import decide
mins = json.load(open('/data/minutes.json'))
candidates = [r for r in mins if <your range>]
probabilities = decide.rank([json.dumps(r['caption'], ensure_ascii=False) for r in candidates], 'Does this caption show <what you are looking for>?', '<what counts>', '<what does not>')
for p, r in sorted(zip(probabilities, candidates), key=lambda pair: -pair[0])[:10]:
    print(f"{p:.2f} {r['clip_id']}")
```

- `decide.rank(texts, question, yes, no, workers=16)`: one request per text, `workers` at a time; returns the yes probability (0 to 1) of each text, in the order of `texts`. `yes` and `no` describe what counts as yes and as no.
- `decide.ask(text, questions)`: one request with your own questions; returns its answers. `questions` maps names to `noul` (`criteria` with `true` and `false`; the answer has `noul`, a probability), `choice` (`criteria` maps labels to descriptions; the answer has `choice` and `probabilities`) or `score` (`criteria` an ordered list of levels; the answer has `score`, the expected level index, and `probabilities`). Several questions can be asked about one text in one request.
- A text is the caption, or part of the caption, of one minute.
- A rate limit or server error is retried up to 5 times; a request that still fails raises and stops the exec: run it again, do not count those texts as "no".
