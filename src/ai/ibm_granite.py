#!/usr/bin/env python3

import json
import threading
import time
import requests

from requests.packages.urllib3.exceptions import InsecureRequestWarning

try:
    from . import config
except ImportError:
   import config

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class GraniteClient(object):
    def __init__(self):
        self._cached_token = None
        self._token_expiry = 0
        self._token_lock = threading.Lock()

    def _get_iam_token(self):
        with self._token_lock:
            if self._cached_token and time.time() < (self._token_expiry - 60):
                return self._cached_token

        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + config.IBM_API_KEY

            response = requests.post(
                config.IBM_AUTH_URL,
                headers=headers,
                data=data,
                verify=False,
                timeout=15,
            )

            if response.status_code == 200:
                token_data = response.json()
                with self._token_lock:
                    self._cached_token = token_data["access_token"]
                    self._token_expiry = time.time() + token_data["expires_in"]
                    return self._cached_token

            print("Token request failed:", response.status_code)
            print("Body:", response.text)
            return None

        except Exception as e:
            print("Token Error:", e)
            return None

    def _post_granite(self, prompt_text, timeout_s=35.0, max_new_tokens=420):
        token = self._get_iam_token()
        if not token:
            print("No IAM token obtained.")
            return None

        payload = {
            "project_id": config.IBM_PROJECT_ID,
            "model_id": config.IBM_MODEL_ID,
            "input": prompt_text,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        }

        try:
            response = requests.post(
                config.IBM_GRANITE_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
                verify=False,
            )

            if response.status_code != 200:
                print("Granite API Error:", response.status_code)
                try:
                    print(response.json())
                except Exception:
                    print(response.text)
                return None

            data = response.json()
            results = data.get("results", [])
            if results and "generated_text" in results[0]:
                return results[0]["generated_text"].strip()

            print("Unexpected Granite response:", data)
            return None

        except Exception as e:
            print("Granite Request Error:", e)
            return None

    def answer_graph_question(self, graph_context):
        prompt_text = (
            "You are an automotive data analysis assistant for engineers.\n"
            "Your job is to analyse vehicle telemetry graphs using only the provided graph context.\n"
            "Write in a technical, engineering-style tone.\n"
            "Be specific about trends, ranges, operating behaviour, relationships, anomalies, and likely interpretations.\n"
            "If the graph is an XY graph, discuss correlation, whether the relationship appears linear or non-linear, and what the slope/direction suggests.\n"
            "If the graph is a time-series graph, discuss overall trend, stability, transient events, spikes, and possible operating phases.\n"
            "Do not invent values or causes not supported by the graph context.\n"
            "If the answer cannot be inferred from the context, say exactly: "
            "'I cannot determine that from the provided graph context.'\n\n"
            "When useful, structure your answer around:\n"
            "1. Main observation\n"
            "2. Supporting evidence from the graph summary\n"
            "3. Engineering interpretation\n"
            "4. Any uncertainty or limitation\n\n"
            "Graph context JSON:\n"
            + json.dumps(graph_context)
        )

        return self._post_granite(prompt_text)
