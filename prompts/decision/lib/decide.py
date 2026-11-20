"""The decision model (convaiinnovations/laya at http://decision/decisions; no key needed) as functions.

    import sys; sys.path.insert(0, '/decision'); import decide
    decide.ask(text, questions) -> the answers of one request (questions as in the API: noul, choice or score)
    decide.rank(texts, question, yes, no, workers=16) -> the yes probability of each text, in the order of texts
"""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'http://decision/decisions'


def ask(text, questions):
    body = json.dumps({'state': text, 'questions': questions}).encode()
    # a rate limit (429) or a server error (5xx) is sent again, up to 5 times as the harness's API clients, after 2, 4, 8, 16 and 32 seconds; any other error raises
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'}, method='POST'), timeout=120) as response:
                return json.loads(response.read())['answers']
        except urllib.error.HTTPError as error:
            if (error.code != 429 and error.code < 500) or attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def rank(texts, question, yes, no, workers=16):
    questions = {'match': {'type': 'noul', 'instructions': question, 'criteria': {'true': yes, 'false': no}}}
    with ThreadPoolExecutor(workers) as pool:
        return [answers['match']['noul'] for answers in pool.map(lambda text: ask(text, questions), texts)]
