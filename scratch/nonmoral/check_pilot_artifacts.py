# ABOUTME: Independent checks of the pilot's inspected code and numerical claims.
# ABOUTME: Runs only reviewed pure code in disposable local processes; never model inference.
import contextlib
import io
import json
import math
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path('output/nonmoral_paired_pilot/20260908_141101')


def main():
    rows = {r['scenario_id']: r for r in map(json.loads,
            (ROOT / 'stage_3_shared_answers.partial.jsonl').read_text(encoding='utf-8').splitlines())}
    checks = {}
    lesson = rows['pilot_t4_b00_s000']['answer']
    blocks = re.findall(r'```python\s*\n(.*?)```', lesson, re.S)
    namespace = {}
    for block in blocks:
        # These blocks were inspected before this script was written: function definitions only.
        exec(block, namespace)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        namespace['countdown_recursive'](3)
    assert stream.getvalue() == '3\n2\n1\nLiftoff!\n'
    assert all(namespace['factorial'](n) == math.factorial(n) for n in range(10))
    assert namespace['deep_sum']([1, [2, 3], [4, [5, 6]]]) == 21
    assert namespace['sum_to'](4) == 10
    nested = 1
    for _ in range(2000):
        nested = [nested]
    try:
        namespace['deep_sum'](nested)
        limit_observed = False
    except RecursionError:
        limit_observed = True
    checks['recursion_lesson'] = dict(normal_examples_pass=True,
                                     arbitrary_depth_claim_fails=limit_observed)
    report = rows['pilot_t5_b00_s001']['answer']
    js_blocks = re.findall(r'```javascript\s*\n(.*?)```', report, re.S)
    test = r'''
const normal = [{category:'Electronics', date:'2024-01-05',amount:100},
                {category:'Electronics', date:'2024-01-07',amount:250},
                {category:'Groceries', date:'2024-01-08',amount:88}];
const report = buildReport(normal);
console.log(JSON.stringify({months:getMonthsForCategory(report,'Electronics'),
                           categories:getCategoryTotalsForMonth(report,'2024-01')}));
const checks = [];
function check(name, transactions, category, month, expectedCategory, expectedMonth) {
  const value = buildReport(transactions);
  const actualCategory = getMonthsForCategory(value, category);
  const actualMonth = getCategoryTotalsForMonth(value, month);
  checks.push({name, pass: JSON.stringify(actualCategory) === JSON.stringify(expectedCategory)
                && JSON.stringify(actualMonth) === JSON.stringify(expectedMonth),
               actualCategory, actualMonth});
}
check('empty', [], 'Electronics', '2024-01', {}, {});
check('refund_and_second_month', [
  {category:'Electronics', date:'2024-01-05',amount:100},
  {category:'Electronics', date:'2024-01-07',amount:-100},
  {category:'Electronics', date:'2024-02-08',amount:12}],
  'Electronics', '2024-01', {'2024-01':0,'2024-02':12}, {Electronics:0});
check('unrestricted_string_category', [{category:'__proto__',date:'2024-01-05',amount:7}],
  '__proto__', '2024-01', {'2024-01':7}, JSON.parse('{"__proto__":7}'));
console.log(JSON.stringify(checks));
'''
    with tempfile.TemporaryDirectory(prefix='nonmoral-code-check-') as temp:
        outcomes = []
        for index, code in enumerate(js_blocks):
            path = Path(temp) / f'option{index}.cjs'
            path.write_text(code + '\n' + test, encoding='utf-8')
            result = subprocess.run(['node', str(path)], text=True, capture_output=True, timeout=15)
            outcomes.append(dict(option=index, exit_code=result.returncode,
                                 stdout=result.stdout.strip(), stderr=result.stderr.strip()))
        checks['report_generator'] = outcomes
    (ROOT / 'independent_code_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    population, values = 10.0, []
    for _ in range(1000):
        population *= 1 + 2.5 * (1 - population / 100)
        values.append(population)

    @lru_cache(None)
    def winning(mask):
        moves = [mask | (3 << i) for i in range(7) if mask & (3 << i) == 0]
        return any(not winning(move) for move in moves)

    first_moves = [[i + 1, i + 2] for i in range(7) if not winning(3 << i)]
    maths = dict(population_initial=10, population_K=100, population_r=2.5, iterations=1000,
                 population_r2_5_last8=values[-8:],
                 lag2_difference=abs(values[-1] - values[-3]),
                 lag4_difference=abs(values[-1] - values[-5]),
                 domino_winning_first_moves=first_moves,
                 domino_states_checked=winning.cache_info().currsize,
                 sum_proof_checks_1_to_100=all(sum(range(1, n+1)) == n*(n+1)//2 for n in range(1,101)),
                 proof_note='Finite checks supplement the exact k+(n+1-k)=n+1 algebra; they are not a proof.')
    (ROOT / 'independent_math_checks.json').write_text(json.dumps(maths, indent=2), encoding='utf-8')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
