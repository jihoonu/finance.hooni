import OpenDartReader
import pandas as pd
import time
import datetime
from rules_config import FINANCIAL_RULES

# ==========================================
# 1. 룰 기반 계정 분류 로직
# ==========================================
def classify_account(account_nm):
    clean_nm = str(account_nm).replace(' ', '')
    for rule in FINANCIAL_RULES:
        inc_words = rule.get('포함단어', [])
        exc_words = rule.get('제외단어', [])
        category = rule.get('구분', '')
        
        has_inc = any(inc in clean_nm for inc in inc_words)
        if has_inc:
            has_exc = any(exc in clean_nm for exc in exc_words)
            if not has_exc:
                return category
    return '기타영업항목'

# ==========================================
# 2. DART 데이터 파싱 헬퍼 함수
# ==========================================
def parse_amount(val):
    if pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace(',', '').replace(' ', '')
    if not s: return 0.0
    if s.startswith('(') and s.endswith(')'): return -float(s[1:-1])
    try: return float(s)
    except ValueError: return 0.0

# ==========================================
# 3. 핵심 재무 분석 로직
# ==========================================
def analyze_structure(df_raw, q_name, prev_state):
    annualize_factor = {"1Q": 4.0, "2Q": 2.0, "3Q": 4.0/3.0, "4Q": 1.0}.get(q_name, 1.0)

    df_bs = df_raw[df_raw['sj_div'] == 'BS'].copy()
    df_is = df_raw[df_raw['sj_div'].isin(['IS', 'CIS'])].copy()
    df_cf = df_raw[df_raw['sj_div'] == 'CF'].copy()

    def get_bs_val(keywords, strict_id=None, col='thstrm_amount'):
        if df_bs.empty: return 0.0
        if strict_id:
            row = df_bs[df_bs['account_id'] == strict_id]
            if not row.empty: return parse_amount(row.iloc[0][col])
            
        clean_nm = df_bs['account_nm'].str.replace(' ', '')
        mask = clean_nm.str.contains('|'.join(keywords), na=False)
        if '부채총계' in keywords:
            mask &= ~clean_nm.str.contains('자본', na=False)
            
        rows = df_bs[mask]
        return parse_amount(rows.iloc[0][col]) if not rows.empty else 0.0

    df_bs['amount'] = df_bs['thstrm_amount'].apply(parse_amount)
    df_bs['category'] = df_bs['account_nm'].apply(classify_account)
    
    total_asset = get_bs_val(['자산총계'], strict_id='ifrs-full_Assets')
    total_debt = get_bs_val(['부채총계'], strict_id='ifrs-full_Liabilities')
    total_equity = get_bs_val(['자본총계'], strict_id='ifrs-full_Equity') or (total_asset - total_debt)

    val_fin_asset = df_bs[df_bs['category'] == '추정재무자산']['amount'].sum()
    val_fin_debt = df_bs[df_bs['category'] == '추정재무부채']['amount'].sum()

    val_op_asset = total_asset - val_fin_asset
    val_op_debt = total_debt - val_fin_debt
    val_net_op_asset = val_op_asset - val_op_debt
    val_net_fin_asset = val_fin_asset - val_fin_debt
    ratio_debt = (total_debt / total_equity * 100) if total_equity else 0

    curr_receivables = get_bs_val(['매출채권'])
    curr_inventory = get_bs_val(['재고자산'])
    controlling_equity = get_bs_val(['지배기업', '지배주주'], strict_id='ifrs-full_EquityAttributableToOwnersOfParent') or total_equity

    if not prev_state:
        prev_equity = get_bs_val(['지배기업', '지배주주'], strict_id='ifrs-full_EquityAttributableToOwnersOfParent', col='frmtrm_amount') or get_bs_val(['자본총계'], col='frmtrm_amount')
        prev_receivables = get_bs_val(['매출채권'], col='frmtrm_amount')
        prev_inventory = get_bs_val(['재고자산'], col='frmtrm_amount')
        prev_noa = val_net_op_asset 
        prev_acc = {}
    else:
        prev_equity = prev_state.get('equity', controlling_equity)
        prev_receivables = prev_state.get('receivables', curr_receivables)
        prev_inventory = prev_state.get('inventory', curr_inventory)
        prev_noa = prev_state.get('noa', val_net_op_asset)
        prev_acc = {} if q_name == '1Q' else prev_state.get('acc', {})

    def get_is_discrete(keywords, strict_id=None, exclude_kws=None):
        if df_is.empty: return 0.0
        def _extract(subset):
            if q_name in ['2Q', '3Q']:
                disc = subset[subset['thstrm_nm'].str.contains('3개월|3 month', na=False, case=False)]
                if not disc.empty: return parse_amount(disc.iloc[0]['thstrm_amount'])
            return parse_amount(subset.iloc[0]['thstrm_amount'])
            
        if strict_id:
            rows = df_is[df_is['account_id'] == strict_id]
            if not rows.empty: return _extract(rows)
            
        clean_nm = df_is['account_nm'].str.replace(' ', '')
        mask = clean_nm.str.contains('|'.join(keywords), na=False)
        rows = df_is[mask].copy()
        
        if exclude_kws:
            rows = rows[~clean_nm.str.contains('|'.join(exclude_kws), na=False)]
        if rows.empty: return 0.0
        return _extract(rows)

    def calc_is_ytd(key, keywords, strict_id=None, exclude_kws=None):
        discrete = get_is_discrete(keywords, strict_id, exclude_kws)
        if q_name in ['2Q', '3Q']:
            return prev_acc.get(key, 0.0) + discrete
        return discrete 

    ytd_revenue = calc_is_ytd('rev', ['매출액', '영업수익', '매출'], strict_id='ifrs-full_Revenue')
    ytd_cogs = calc_is_ytd('cogs', ['매출원가', '영업비용'], strict_id='ifrs-full_CostOfSales')
    ytd_gross_profit = calc_is_ytd('gp', ['매출총이익'], strict_id='ifrs-full_GrossProfit')
    ytd_op_income = calc_is_ytd('op', ['영업이익', '영업손실'], strict_id='ifrs-full_OperatingIncomeLoss')
    ytd_ebt = calc_is_ytd('ebt', ['법인세비용차감전', '세전이익'])
    ytd_tax = calc_is_ytd('tax', ['법인세비용', '법인세'], strict_id='ifrs-full_IncomeTaxExpenseContinuingOperations', exclude_kws=['차감전'])
    
    ytd_net_income = calc_is_ytd('ni', ['지배'], strict_id='ifrs-full_ProfitLossAttributableToOwnersOfParent')
    if ytd_net_income == 0: 
        ytd_net_income = calc_is_ytd('ni', ['당기순이익', '당기순손실'], strict_id='ifrs-full_ProfitLoss')

    def get_cf_ytd(keywords, strict_id=None, exclude_kws=None):
        if df_cf.empty: return 0.0
        if strict_id:
            rows = df_cf[df_cf['account_id'] == strict_id]
            if not rows.empty: return parse_amount(rows.iloc[0]['thstrm_amount'])
            
        clean_nm = df_cf['account_nm'].str.replace(' ', '')
        mask = clean_nm.str.contains('|'.join(keywords), na=False)
        rows = df_cf[mask].copy()
        if exclude_kws:
            rows = rows[~clean_nm.str.contains('|'.join(exclude_kws), na=False)]
        if rows.empty: return 0.0
        return parse_amount(rows.iloc[0]['thstrm_amount'])

    def get_cf_group_sum(include_kws, action_kws, exclude_kws=None):
        if df_cf.empty: return 0.0
        nm = df_cf['account_nm'].str.replace(' ', '').str.lower()
        mask = nm.str.contains('|'.join(include_kws), na=False) & nm.str.contains('|'.join(action_kws), na=False)
        if exclude_kws:
            mask &= ~nm.str.contains('|'.join(exclude_kws), na=False)
        return sum(abs(parse_amount(v)) for v in df_cf.loc[mask, 'thstrm_amount'])

    ytd_cfo = get_cf_ytd(['영업활동'], strict_id='ifrs-full_CashFlowsFromUsedInOperatingActivities')
    ytd_cfi = get_cf_ytd(['투자활동'], strict_id='ifrs-full_CashFlowsFromUsedInInvestingActivities')
    ytd_cff = get_cf_ytd(['재무활동'], strict_id='ifrs-full_CashFlowsFromUsedInFinancingActivities')

    tan_kws, int_kws = ['유형자산', '기계', '설비', '토지', '건물'], ['무형자산', '개발비', '소프트웨어', '특허']
    acq_kws, disp_kws = ['취득', '증가', '지출'], ['처분', '매각', '감소']
    ex_kws = ['투자부동산', '금융', '대체']

    ppe_acq = abs(get_cf_ytd([], strict_id='ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment')) or get_cf_group_sum(tan_kws, acq_kws, ex_kws + disp_kws)
    ppe_disp = abs(get_cf_ytd([], strict_id='ifrs-full_ProceedsFromSalesOfPropertyPlantAndEquipment')) or get_cf_group_sum(tan_kws, disp_kws, ex_kws + acq_kws)
    int_acq = abs(get_cf_ytd([], strict_id='ifrs-full_PaymentsToAcquireIntangibleAssets')) or get_cf_group_sum(int_kws, acq_kws, ex_kws + disp_kws)
    int_disp = abs(get_cf_ytd([], strict_id='ifrs-full_ProceedsFromSalesOfIntangibleAssets')) or get_cf_group_sum(int_kws, disp_kws, ex_kws + acq_kws)

    net_ppe_acq = ppe_acq - ppe_disp
    net_int_acq = int_acq - int_disp
    ytd_capex = net_ppe_acq + net_int_acq 
    ytd_fcff = ytd_cfo - ytd_capex

    avg_receivables = (prev_receivables + curr_receivables) / 2 if prev_receivables else curr_receivables
    turnover_recv = (ytd_revenue * annualize_factor / avg_receivables) if avg_receivables > 0 else 0
    days_recv = (365 / turnover_recv) if turnover_recv > 0 else 0

    avg_inventory = (prev_inventory + curr_inventory) / 2 if prev_inventory else curr_inventory
    turnover_inv = (abs(ytd_cogs) * annualize_factor / avg_inventory) if avg_inventory > 0 else 0
    days_inv = (365 / turnover_inv) if turnover_inv > 0 else 0

    gp_margin = (ytd_gross_profit / ytd_revenue * 100) if ytd_revenue > 0 else 0
    op_margin = (ytd_op_income / ytd_revenue * 100) if ytd_revenue > 0 else 0
    
    effective_tax_rate = ytd_tax / ytd_ebt if ytd_ebt > 0 else 0.22
    if not (0 <= effective_tax_rate <= 1.0): effective_tax_rate = 0.22

    avg_equity = (prev_equity + controlling_equity) / 2 if prev_equity else controlling_equity
    roe = (ytd_net_income * annualize_factor / avg_equity * 100) if avg_equity > 0 else 0

    nopat = ytd_op_income * (1 - effective_tax_rate)
    avg_noa = (prev_noa + val_net_op_asset) / 2 if prev_noa else val_net_op_asset
    roic = (nopat * annualize_factor / avg_noa * 100) if avg_noa > 0 else 0

    # ----------------------------------------
    # [Step 5] 화면 표출 데이터
    # ----------------------------------------
    data_labels = [
        '1. 추정영업자산', '2. 추정재무자산', '3. 자산총계', 
        '4. 추정영업부채', '5. 추정재무부채', '6. 부채총계', '7. 순자산 (자본총계)', 
        '8. 추정순영업자산', '9. 추정순재무자산', '10. 부채비율 (%)', 
        '11. 매출액', '  - 매출채권회전율 (회)', '  - 매출채권회전기간 (일)', 
        '12. 매출원가', '  - 재고자산회전율 (회)', '  - 재고자산회전기간 (일)', 
        '13. 매출총이익', '  - 총이익률 (%)', '14. 영업이익', '  - 영업이익률 (%)', 
        '15. 세전이익', '16. 법인세비용', '  - 실효세율 (추정 %)', '17. 당기순이익', 
        '18. ROE (%)', 
        '  [검증] 연환산 지배순이익',
        '  [검증] 평균 지배자본', 
        '19. ROIC (%)',
        '20. 영업활동현금흐름', '21. 투자활동현금흐름', '  - 유형자산순취득액', '  - 무형자산순취득액', '  - 자본적지출(CAPEX)',
        '22. 재무활동현금흐름', '  - FCFF'
    ]
    
    def scale(v): return v / 1000000 if v else 0

    data_values = [
        scale(val_op_asset), scale(val_fin_asset), scale(total_asset), 
        scale(val_op_debt), scale(val_fin_debt), scale(total_debt), scale(total_equity), 
        scale(val_net_op_asset), scale(val_net_fin_asset), ratio_debt, 
        
        scale(ytd_revenue), turnover_recv, days_recv, 
        scale(ytd_cogs), turnover_inv, days_inv, 
        scale(ytd_gross_profit), gp_margin, scale(ytd_op_income), op_margin, 
        scale(ytd_ebt), scale(ytd_tax), effective_tax_rate*100, scale(ytd_net_income), 
        roe, 
        scale(ytd_net_income * annualize_factor), 
        scale(avg_equity),
        roic, 
        
        scale(ytd_cfo), scale(ytd_cfi), scale(net_ppe_acq), scale(net_int_acq), scale(ytd_capex), scale(ytd_cff), scale(ytd_fcff)
    ]
    
    current_state = {
        'equity': controlling_equity,
        'receivables': curr_receivables, 
        'inventory': curr_inventory,
        'noa': val_net_op_asset,
        'acc': {
            'rev': ytd_revenue,
            'cogs': ytd_cogs,
            'gp': ytd_gross_profit,
            'op': ytd_op_income,
            'ebt': ytd_ebt,
            'tax': ytd_tax,
            'ni': ytd_net_income
        }
    }
    
    return data_labels, data_values, current_state
# ==========================================
# 4. 분석 실행기 루프 (수정됨)
# ==========================================
def run_financial_analysis(api_key, company_name, start_year, progress_callback=None):
    dart = OpenDartReader(api_key)
    
    # [수정 1] 사용자가 입력한 이름으로 DART에 등록된 기업 찾기 검증
    company = dart.find_by_name(company_name)
    if company is None or company.empty:
        print(f"오류: DART에서 '{company_name}'을(를) 찾을 수 없습니다. 공식 명칭이나 종목코드를 확인하세요.")
        return None
    
    # 여러 개가 검색될 수 있으므로 첫 번째(가장 정확한 상장사 위주) 회사의 공식 명칭이나 종목코드 사용
    # DART 고유번호(corp_code)를 사용하는 것이 가장 안전합니다.
    target_corp_code = company.iloc[0]['corp_code'] 

    reprt_codes = {'11013': '1Q', '11012': '2Q', '11014': '3Q', '11011': '4Q'}

    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month

    valid_periods = []
    valid_periods.append((start_year - 1, '11011', '4Q'))

    for year in range(start_year, current_year + 1):
        for code, q_name in reprt_codes.items():
            if year == current_year:
                if q_name == '1Q' and current_month <= 3: continue
                if q_name == '2Q' and current_month <= 6: continue
                if q_name == '3Q' and current_month <= 9: continue
                if q_name == '4Q': continue 
            valid_periods.append((year, code, q_name))

    total_steps = len(valid_periods)
    current_step = 0
    final_dict = {}
    row_labels = []
    prev_state = {}

    for year, code, q_name in valid_periods:
        current_step += 1
        period_name = f"{str(year)[-2:]}년 {q_name}"

        if progress_callback:
            progress_callback(current_step, total_steps, f"{period_name} 데이터 수집 중...")

        try:
            # [수정 2] company_name 대신 target_corp_code(고유번호) 사용
            df_all = dart.finstate_all(target_corp_code, year, reprt_code=code)
            
            if df_all is not None and not df_all.empty:
                # analyze_structure 함수는 df_all에서 데이터를 추출하므로 그대로 유지
                labels, values, current_state = analyze_structure(df_raw=df_all, q_name=q_name, prev_state=prev_state)

                # 주의: 리턴받은 current_state를 다시 prev_state에 넣는 로직은
                # 원본 코드 구조상 analyze_structure 안에서 current_state를 만들고 반환해야 작동합니다.
                # 올려주신 코드에는 analyze_structure 함수의 return 값에 current_state가 빠져있습니다! (아래 참고)
                
                final_dict[period_name] = values
                if not row_labels: 
                    row_labels = labels

                prev_state = current_state

            time.sleep(0.5) 
        except Exception as e:
            # [수정 3] 에러가 났을 때 원인을 파악하기 위해 에러 로그 출력
            print(f"[{period_name}] 데이터 수집 실패: {e}")
            pass 

    if not final_dict:
        return None

    result_df = pd.DataFrame(final_dict, index=row_labels)
    return result_df