# ABOUTME: Offline integrity, prompt-constraint and behavioral checks for reviewed cached answers.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/check_cached_answers.py
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from omegaconf import OmegaConf
from run import readrows, write, digest

cfg = OmegaConf.load(Path(__file__).with_name('cached_answer_review.yaml'))
out = Path(cfg.output)
prior = Path(cfg.prior_output)
before = readrows(prior/'dataset/mixture.jsonl')
after = readrows(out/'dataset/mixture.jsonl')
approved = {(int(d.row), int(d.turn)): d for d in cfg.review.approved}
failures = {(r['row'], r['turn']) for r in readrows(out/'dataset/failed_replacements.jsonl')}
old_failures = {(r['row'], r['turn']) for r in readrows(prior/'failed_replacements.jsonl')}
assert len(before) == len(after) == 10000
assert set(approved).isdisjoint(failures) and set(approved) | failures == old_failures
content = {}
for i, (a, b) in enumerate(zip(before, after)):
    restored = copy.deepcopy(b)
    for j, (old, new) in enumerate(zip(a['messages'], b['messages'])):
        if (i, j) in approved:
            assert old.get('reasoning_content') and j == len(a['messages']) - 1, (i, 'future context needs review')
            d = approved[i, j]
            r = json.loads((prior/f'backfill_receipts/{i}_{j}_{d.attempt}.json').read_text(encoding='utf-8'))
            assert new['content'] == r['response']['content']
            assert new['reasoning_content'] == r['response']['reasoning_content']
            for field in ['content', 'reasoning_content']:
                restored['messages'][j][field] = old[field]
            content[i] = new['content'].strip()
        else:
            assert old == new
    assert restored == a

checks = []
def check(row, condition, description):
    assert condition, (row, description)
    checks.append({'row': row, 'check': description, 'passed': True})

def text(row):
    # Markdown emphasis is presentation; do not rewrite the cached answer itself.
    return content[row].replace('**', '').strip()

def words(row, word):
    return len(re.findall(r'\b'+re.escape(word)+r'\b', text(row), re.I))

for row in [357, 1552, 2044, 2120, 5778, 6278, 6525, 6538, 7663, 8989, 1121, 1155, 9678, 7726]:
    check(row, ',' not in content[row], 'No commas')
for row in [1893, 2728, 3357, 3957, 4204, 4644, 5232, 5764, 5815, 7032, 8423, 8449, 7726]:
    check(row, content[row] == content[row].upper(), 'All letters uppercase')
for row in [909, 2120, 3353]:
    check(row, content[row] == content[row].lower(), 'All letters lowercase')
for row, limit in {909:150, 1517:50, 4030:150, 4529:150, 9678:30}.items():
    check(row, len(content[row].split()) <= limit, f'At most {limit} whitespace-delimited words')
for row, counts in {145:{'admire':4,'bravery':2}, 1224:{}, 1616:{'ergonomic':4,'aesthetic':3},
    5404:{'vibrant':4,'texture':3}, 5885:{'bond':4,'second chance':3},
    6407:{'I want to make a difference in our community':2},7031:{'aguas':2},
    7322:{'home':4,'family':3},8449:{'MEDICATION SAFETY':2,'MANAGEMENT':2},
    8855:{'education':4,'past':3},9383:{'transcendence':2}}.items():
    for word, minimum in counts.items():
        check(row, words(row, word) >= minimum, f'At least {minimum} occurrences of {word}')
check(1224, text(1224).lower().count('e') >= 100, 'At least 100 letter e occurrences')
for row, end in {145:'Together, we will balance the scales of risk and reward.',
    1224:'In the end, the words were truly hers.',4237:'And thus, their journey in the realm of imagination began.',
    5885:'Together, we will cherish every moment, hand in hand.',
    5986:'You are, and always will be, our greatest blessing.',
    7322:'We hope you find comfort here and feel at ease.',
    8855:'The journey continues with every lesson I learn.',
    8922:'The phase angle between the voltage and current is crucial for optimizing power transfer.',
    3076:"He knew the race wasn't over yet.",
    2526:'Together, we can build bridges that strengthen our community for generations to come.'}.items():
    check(row, text(row).endswith(end), 'Required final sentence (after removing Markdown emphasis)')
for row, count in {1616:3,4237:3,4443:4,4532:2,5008:4,7031:1,8100:2}.items():
    check(row, len(re.split(r'\n\s*\n', text(row))) == count, f'{count} prose paragraphs')
check(4443, re.split(r'\n\s*\n', text(4443))[2].startswith('Moreover'), 'Third paragraph starts Moreover')
check(4443, text(4443).endswith('healing'), 'Ends with healing')
check(5008, text(5008).startswith('Growing') and re.split(r'\n\s*\n', text(5008))[2].rstrip().endswith('harmony'), 'Growing start and third paragraph ends harmony')
for row, delimiter in {1115:'***',7016:'---'}.items():
    check(row, len(content[row].split(delimiter)) == 3, 'Three sections with required separators')
for row, forbidden in {586:['taxes','insurance','retirement'],2128:['discount','affordable'],
    7453:['inspiration','guidance','teaching'],960:['delay','slow'],
    5794:['trust','politicians','vote'],6771:['plot','character','story'],
    3076:['accident','fear'],7337:['death','court']}.items():
    check(row, not any(words(row, w) for w in forbidden), 'Excluded words absent')
for row in [553, 1249, 6444, 7337, 9930]:
    check(row, text(row).startswith('"') and text(row).endswith('"'), 'Entire response in double quotes')
for row, minimum in {102:4,4952:3,4532:3,3076:3}.items():
    count = len(re.findall(r'\[[^\[\]]+\]', text(row)))
    check(row, count == 3 if row == 3076 else count >= minimum, 'Required placeholder count')
for row in [5967,7905,3353,2418]:
    cleaned = re.sub(r'^```json\s*|\s*```$', '', content[row]).strip()
    obj = json.loads(cleaned)
    check(row, isinstance(obj, (dict,list)), 'Valid JSON payload (optional Markdown fence)')
    if row == 7905:
        check(row, set(obj) == {'sentence_1','sentence_2','sentence_3'} and all(
            len(re.findall(r'\badaptation\b', v)) == 2 for v in obj.values()), 'Three fields each containing adaptation twice')
    if row == 5967:
        check(row, set(obj) == {'bullet_1','bullet_2','bullet_3'}, 'Three required JSON fields')
check(6438, text(6438).startswith(before[6438]['messages'][0]['content']), 'Prompt repeated verbatim')
check(9005, len(re.findall(r'^\d\.',text(9005),re.M)) == 4 and words(9005,'Squad') == 8, 'Four team names, each with two Squad tokens')
check(2741, all(f'Stanza {i}' in text(2741) for i in range(1,6)), 'Five marked stanzas')
check(7032, len(content[7032].splitlines()) == 3, 'Three-line chant')
check(653, len(re.findall(r'[.!?](?:\s|$)',text(653))) == 4, 'Four sentences')
check(8922, len(re.findall(r'[.!?](?:\s|$)',text(8922))) == 5, 'Five sentences')
check(9383, len(re.findall(r'[.!?](?:\s|$)',text(9383))) == 5, 'Five sentences')
check(2120, all('translation' in s and 'accuracy' in s and any(w.count('e')>=3 for w in s.split())
    for s in text(2120).split('.') if s.strip()), 'Each sentence has translation, accuracy and a word with three e letters')

# Execute only the first, fully inspected Python block in the approved simple-code cases.
# Reject unexpected capabilities before execution; no arbitrary dataset code is evaluated.
code_rows = [1412,1886,2609,3170,4658,4938,6098,6411,6692,7266,8450,8551,8795,8883,9354,9921]
environments = {}
for row in code_rows:
    code = re.search(r'```python\s*\n(.*?)```', content[row], re.S).group(1)
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name in {'math','typing','__future__'} for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in {'math','typing','__future__'}
        if isinstance(node, ast.Name):
            assert node.id not in {'eval','exec','open','compile','globals','locals','input','__import__'}
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith('__') or node.attr == '__init__'
    env = {}
    exec(compile(tree, f'<reviewed-cached-row-{row}>', 'exec'), env)
    environments[row] = env

def call(row, name, args, expected):
    result = environments[row][name](*args)
    check(row, result == expected, f'{name} behavioral example')

for args, expected in [(([3,1,2,2],[2,3,3]),[2,3]),(([],[1]),[])]: call(1886,'unique_sorted_intersection_list',args,expected)
for args, expected in [((5,5),True),((5,5.0),False),(([1,2],[1,2]),True)]: call(2609,'same_value_and_type',args,expected)
for value, expected in [('1234567890',''),('!@#$%^&*()',''),('Hello, World!','HelloWorld'),('123 ABC','ABC'),(None,''),('é1','é')]: call(3170,'filter_alpha',(value,),expected)
call(1412,'assert_positive',(0,),None)
call(4658,'sample_mean',([1,2,3],),2.0)
call(4658,'sample_mean',([-4,2],),-1.0)
call(4938,'generate_reversed_list',([1,2,3,4,5],),[5,4,3,2,1])
original = [1,2,3]
result = environments[4938]['generate_reversed_list'](original)
check(4938, result is not original and original == [1,2,3], 'New list without mutation')
call(6098,'reverse_file',(['a','b'],),['b','a'])
call(6098,'reverse_file',([],),[])
call(6411,'ensure_in_range',(1,),None)
call(6411,'ensure_in_range',(1000,),None)
call(6692,'cone_volume',(0,3),0.0)
check(6692, math.isclose(environments[6692]['cone_volume'](10,3),30*math.pi), 'Cone volume formula')
for args, expected in [((8,9,'/'),8/9),((6,7,'*'),42),((2,3,'+'),5),((4,5,'-'),-1),((2,3,'unknown'),None)]: call(7266,'perform_arithmetic_operation',args,expected)
for args, expected in [(('Hello, world!',5),'Hello'),(('Hello, world!',100),'Hello, world!'),(('',1),'')]: call(8450,'truncate_string',args,expected)
call(8551,'string_formatting',([1,2,3],),['Number 0 is: 1','Number 1 is: 2','Number 2 is: 3'])
for x,y in [(0,.1),(1,.2),(99,10.0)]: call(8795,'calculate_y_axis_value',(x,),y)
for x,y in [('Hello!',True),('Hello.',True),('Hello,',False),('Hello?',True),('',False)]: call(8883,'ends_with_punctuation',(x,),y)
for x,y in [(['a','b'],True),(['a',''],False),(['a',None],False),([1],False),([],True)]: call(9354,'all_nonempty_strings',(x,),y)
for x,y in [(0,1),(1,1),(5,120)]: call(9921,'calculate_factorial',(x,),y)
for row,name,args,error in [(1412,'assert_positive',(-1,),AssertionError),(4658,'sample_mean',([],),AssertionError),
    (4658,'sample_mean',((1,2),),AssertionError),(6411,'ensure_in_range',(0,),ValueError),
    (9921,'calculate_factorial',(1.5,),TypeError),(9921,'calculate_factorial',(-1,),ValueError)]:
    try: environments[row][name](*args)
    except error: check(row, True, f'{name} specified error behavior')
    else: raise AssertionError((row, 'expected exception'))

report = {'passed': True, 'checked_pairs': len(approved), 'local_checks': len(checks),
    'dataset_file_sha256': hashlib.sha256((out/'dataset/mixture.jsonl').read_bytes()).hexdigest(),
    'all_10000_rows_verified': True, 'all_unselected_messages_preserved': True,
    'all_replacements_are_terminal_turns': True, 'checks': checks,
    'limits': 'Local checks and Codex review are not an exhaustive formal proof of answer correctness.'}
write(out/'dataset/cached_answer_checks.json', report)
print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))
