from adaptivity.level_assessor import run_assessment
from database.db import get_quiz_scores, get_user

u = get_user('haseeb')
print('current_level:', u['current_level'])
print('last 5 scores:', get_quiz_scores('haseeb', 5))

result = run_assessment('haseeb', u['current_level'])
print('assessment result:', result)