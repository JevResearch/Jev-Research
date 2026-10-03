#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from jev_observatory.expanded_dispatch import dispatch

def main():
    import os
    from jev_observatory.redact import Redactor

    p = argparse.ArgumentParser()
    p.add_argument('--planpath', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ledger', required=True)
    p.add_argument('--cap', type=float, default=20)
    p.add_argument('--key-environment', default='TYPESAFE_API_KEY')
    p.add_argument('--concurrency', type=int, default=1,
                   help='Requested bound; current executor is serial.')
    a = p.parse_args()
    key = os.environ.get(a.key_environment)
    try:
        summary = dispatch(a.planpath, a.output, a.ledger, a.cap,
                           key, a.concurrency)
    except Exception as exc:
        print(json.dumps({'error': Redactor([key] if key else []).exception(exc)}),
              file=sys.stderr)
        return 1
    summary['effective_concurrency'] = 1
    print(json.dumps(summary))
    return int(bool(summary['stopped'] or summary['errors'] or summary['unsent']))

if __name__ == '__main__':
    raise SystemExit(main())
