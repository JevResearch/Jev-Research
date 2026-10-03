"""Author-requested chart axes, placement and counterpoint hierarchy."""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'report'))
import language_graph


def test_grouped_bars_have_correct_geometry_and_scores():
    svg = language_graph.render(ROOT)
    tree = ET.fromstring(svg)
    ns = {'s': 'http://www.w3.org/2000/svg'}
    bars = tree.findall('.//s:rect[@class="bar"]', ns)
    assert len(bars) == 12
    aggregate = json.loads((ROOT / 'data_report/expanded_20261003/'
                             'multilingual_heldout_aggregates.json').read_text())
    by_lang = {r['language']: r for r in aggregate['results']}
    for i, code in enumerate(['en', 'ar', 'zh', 'de', 'ru', 'es']):
        first, second = bars[i * 2:i * 2 + 2]
        assert float(first.attrib['x']) + float(first.attrib['width']) < float(second.attrib['x'])
        for bar, field in [(first, 'jev_accuracy'), (second, 'qwen_accuracy')]:
            height = float(bar.attrib['height'])
            top = float(bar.attrib['y'])
            assert abs(height - 535 * by_lang[code][field]) < .1
            assert abs(top + height - 605) < .2
            assert 100 <= float(bar.attrib['x']) < 1210
            assert 70 <= top <= 605
            assert f"{100 * by_lang[code][field]:.2f}%" in svg
    assert not tree.findall('.//s:circle', ns)
    assert 'Accuracy (%)' in svg
    assert '#language-comparison-chart text' in svg


def test_matched_sections_follow_costs_and_precede_identity_probes():
    html = (ROOT / 'report/index.html').read_text()
    assert 'arch-ancestry' not in html
    assert html.index('id="pareto"') < html.index('id="matched-comparisons"')
    assert html.index('id="bench-language"') < html.index('id="bench-politics"')
    assert html.index('id="bench-politics"') < html.index('id="probes"')
    assert 'Each request supplied a premise and a hypothesis.' in html
    assert 'The box contains exactly two apples' in html
    assert 'task instructions and answer labels were English in every language' in html


def test_counterpoint_evidence_subsections_and_author_cut():
    cp = (ROOT / 'counterpoint/index.html').read_text()
    assert 'The current article itself says' not in cp
    for anchor in ['counter-tokenizer', 'counter-language', 'counter-politics']:
        assert re.search(r'<h3 id="' + anchor + r'">[^<]+</h3>', cp)
    for anchor in ['counter-ancestry', 'counter-similarities', 'counter-computation']:
        assert re.search(r'<h2 id="' + anchor + r'">[^<]+</h2>', cp)
