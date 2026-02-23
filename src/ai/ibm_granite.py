#!/usr/bin/env python

import requests
import time
import json
import threading
import config

# Disables SSL warnings for Python 2.7.
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class GraniteClient(object):
    def __init__(self):
        self._cached_token = None
        self._token_expiry = 0
        self._token_lock = threading.Lock()

    def _get_iam_token(self):
        # Retrieves or refreshes the IAM token.
        with self._token_lock:
            if self._cached_token and time.time() < (self._token_expiry - 60):
                return self._cached_token

        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + config.IBM_API_KEY
            
            response = requests.post(config.IBM_AUTH_URL, headers=headers, data=data, verify=False)
            
            if response.status_code == 200:
                json_resp = response.json()
                with self._token_lock:
                    self._cached_token = json_resp["access_token"]
                    self._token_expiry = time.time() + json_resp["expires_in"]
                return self._cached_token
        except Exception as e:
            print "Token Error:", e
        return None

    def _worker_logic(self, metrics):
        # This is the actual logic that runs inside the thread.
        token = self._get_iam_token()
        if not token: return

        try:
            prompt_text = (
                "You are a professional Race Engineer. "
                "Analyze this telemetry JSON: " + json.dumps(metrics) + ". "
                "Current state: Speed (speedX) is in km/h. TrackPos is -1 to 1 (0 is center). "
                "Give 1 sentence of status starting with the Track Position value, then Speed. Then give 1 driving instruction."
            )

            payload = {
                "project_id": config.IBM_PROJECT_ID,
                "model_id": config.IBM_MODEL_ID,
                "input": prompt_text,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 150,
                    "min_new_tokens": 1
                }
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token
            }

            response = requests.post(config.IBM_GRANITE_URL, headers=headers, json=payload, timeout=2.0, verify=False)

            if response.status_code == 200:
                print "\n>> AI Race Engineer:", response.json()['results'][0]['generated_text'].strip()
                print "--------------------------------------------------"
            else:
                print "Granite API Error:", response.status_code

        except Exception:
            pass # Fails silently to keep car driving, prevents crashes.

    def send_async_analysis(self, metrics):
        # Starts the analysis in a background thread.
        t = threading.Thread(target=self._worker_logic, args=(metrics,))
        t.daemon = True
        t.start()