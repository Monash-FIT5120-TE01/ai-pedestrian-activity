"""Small read-only JSON API for a separately built frontend."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

if __package__:
    from .predict import ForecastService, add_paths
else:
    from predict import ForecastService, add_paths

LOG = logging.getLogger(__name__)


def make_handler(service, allowed_origins):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            raw = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Vary', 'Origin')
            origin = self.headers.get('Origin')
            if origin in allowed_origins:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_json(200, {})

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                query = parse_qs(parsed.query, keep_blank_values=True)
                if any(len(v) != 1 for v in query.values()):
                    raise ValueError('Query parameters must occur once.')
                if parsed.path == '/health':
                    return self.send_json(200, dict(status='ok', service='street-forecast'))
                if parsed.path == '/api/sensors':
                    return self.send_json(200, dict(sensors=list(service.sensors.values())))
                if parsed.path == '/api/input-status':
                    if set(query)-{'sensor_id'}:
                        raise ValueError('Supported input-status query parameter: sensor_id.')
                    sid = int(query['sensor_id'][0]) if 'sensor_id' in query else None
                    return self.send_json(200, service.observation_status(sid))
                if parsed.path != '/api/forecast':
                    return self.send_json(404, dict(error='not_found'))
                if set(query)-{'sensor_id', 'hours'}:
                    raise ValueError('Supported query parameters: sensor_id, hours.')
                hours = int(query['hours'][0]) if 'hours' in query else None
                if hours is not None and not 1 <= hours <= 8760:
                    raise ValueError('hours must be between 1 and 8760.')
                if 'sensor_id' in query:
                    result = service.forecast(int(query['sensor_id'][0]), hours)
                    return self.send_json(200 if result['status'] == 'available' else 503, result)
                self.send_json(200, service.forecast_all(hours))
            except (ValueError, KeyError) as exc:
                LOG.warning('Invalid request or input snapshot: %s', exc)
                self.send_json(400, dict(error='invalid_request_or_inputs', reason=str(exc)))
            except Exception:
                LOG.exception('Forecast request failed')
                self.send_json(500, dict(error='forecast_failed', reason='Check backend logs and artifact/input files.'))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--origin', action='append',
                        help='Allowed frontend origin; repeat for multiple origins.')
    args = parser.parse_args()
    service = ForecastService(args.data, args.artifacts, args.weather, args.selection,
                              fallback_seasonal=args.fallback_seasonal)
    origins = args.origin or ['http://localhost:5174', 'http://127.0.0.1:5174']
    server = HTTPServer((args.host, args.port), make_handler(service, origins))
    print(f'Forecast API: http://{args.host}:{args.port}/api/forecast', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
