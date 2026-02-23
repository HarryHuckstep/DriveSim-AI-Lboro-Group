#!/usr/bin/env python

def parse_raw_telemetry(buf):
    # Parses raw TORCS string format "(angle 0)(gear 1)..." into a dictionary.
    data = {}
    try:
        # Strips whitespace and splits by ')(' to separate data points.
        items = buf.strip().split(')(')
        
        for item in items:
            # Cleans parenthesis.
            clean_item = item.replace('(', '').replace(')', '')
            parts = clean_item.split(' ')
            
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1]
                
                # Filters only the data we want to send to the AI.
                if key in ['gear', 'rpm', 'speedX', 'trackPos', 'damage']:
                    try:
                        data[key] = float(val)
                    except ValueError:
                        pass
    except Exception:
        pass
        
    return data