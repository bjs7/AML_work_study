# %%
import sys
import pandas as pd
from pathlib import Path

sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')

from analysis_functions import df_to_latex_table
from data.relevant_banks import load_relevant_banks

TABLES_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/heterogeneity')

SIZE, IR = 'small', 'HI'

FILTER_ROWS = [
    ('no_top10',    'H1 (no top-10)'),
    ('no_top1',     'H2 (no top-1)'),
    ('no_bottom10', 'H3 (no bottom-10)'),
    ('no_bottom5pct', r'H4 (no bottom-5\%)'),
]


def build_bank_filter_table(stats, eval_mode):
    rows = []
    for filter_name, label in FILTER_ROWS:
        s = stats[filter_name]
        rows.append({
            'ID': label,
            r'$|\mathcal{K}_{\mathrm{tr}}|$': s['train_banks_remaining'],
            'Train cov. (\\%)': round(s['data_pct_left'] * 100, 2),
            'Illicit cov. (\\%)': round(s['laundering_pct_left'] * 100, 2),
            'Vali unreachable (\\%)': round(s.get('vali_unreachable_illicit_pct', float('nan')) * 100, 2),
            'Test unreachable (\\%)': round(s.get('test_unreachable_illicit_pct', float('nan')) * 100, 2),
        })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / f'bank_filter_coverage_{eval_mode}.tex'
    df_to_latex_table(df, out_path)
    print(f"Saved: {out_path}")


# %%

class _FakeDataParser:
    def __init__(self, size, ir):
        self.size = size
        self.ir = ir


data_parser = _FakeDataParser(SIZE, IR)
relevant_banks = load_relevant_banks(data_parser)

build_bank_filter_table(relevant_banks['bank_filter_system'],     'system')
build_bank_filter_table(relevant_banks['bank_filter_comparable'], 'comparable')
