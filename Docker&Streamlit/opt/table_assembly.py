import re
import pandas as pd


def assembly(lst):
    df = pd.DataFrame(columns=['date', 'donation_type', 'is_paid'])
    i = 0
    while i < len(lst):
        date = None
        donation_type = None
        is_paid = None
        
        if re.match(
            r'^[\W_]?\d{2}[\W_]?\d{2}[\W_]?(\d{2}|\d{4})[\W_]?$',
            lst[i]
            ) and i+2 < len(lst) and re.match(r'\D+', lst[i+1]):
            date_parts = re.findall(r'(\d{2})[\W_]?(\d{2})', lst[i])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            donation_type = lst[i+1]
            is_paid = lst[i+2]
            i += 3

        elif re.match(r'^[\W_]?\d{2}[\W_]?$', lst[i]) and i+3 < len(lst) \
          and re.match(r'^[\W_]?\d{2}[\W_]?(\d{2}|\d{4})[\W_]?$', lst[i+1]) \
          and re.match(r'\D+', lst[i+2]):
            date_parts = re.findall(r'(\d{2})[\W_]*(\d{2})',lst[i]+lst[i+1])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+1])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            donation_type = lst[i+2]
            is_paid = lst[i+3]
            i += 4

        elif re.match(r'^[\W_]?\d{2}[\W_]?\d{2}[\W_]?$', lst[i]) \
          and i+3 < len(lst) \
          and re.match(r'^[\W_]?(\d{2}|\d{4})[\W_]?$', lst[i+1]) \
          and re.match(r'\D+', lst[i+2]):
            date_parts = re.findall(r'(\d{2})[\W_]?(\d{2})', lst[i])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+1])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            donation_type = lst[i+2]
            is_paid = lst[i+3]
            i += 4

        elif re.match(r'^[\W_]?\d{2}[\W_]?$', lst[i]) and i+4 < len(lst) \
          and re.match(r'^[\W_]?\d{2}[\W_]?$', lst[i+1]) \
          and re.match(r'^[\W_]?(\d{2}|\d{4})[\W_]?$', lst[i+2]) \
          and re.match(r'\D+', lst[i+3]):
            date_parts = re.findall(r'(\d{2})[\W_]*(\d{2})',lst[i]+lst[i+1])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+2])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            donation_type = lst[i+3]
            is_paid = lst[i+4]
            i += 5
        
        if donation_type:
            if 't' in donation_type.lower():
                donation_type = 'Тромбоциты'
            elif 'k' in donation_type.lower() \
              or 'w' in donation_type.lower() \
              or 'p' in donation_type.lower():
                donation_type = 'Цельная кровь'
            elif 'n' in donation_type.lower() \
              or 'm' in donation_type.lower() \
              or 'v' in donation_type.lower():
                donation_type = 'Плазма'
            else:
                donation_type = None
        
        if is_paid:
            if 0 < len(re.findall(r'[a-zA-Z0-9]', is_paid)) < 4:
                is_paid = 'Безвозмездно'
            elif 3 < len(re.findall(r'[a-zA-Z0-9]', is_paid)) < 6 \
              and ('n' in is_paid.lower() or 't' in is_paid.lower()):
                is_paid = 'Платно'
            elif len(re.findall(r'[a-zA-Z0-9]', is_paid)) > 14 \
              or len(re.findall(r'[a-zA-Z0-9]', is_paid)) == 0 \
              or len(re.findall(r'\d', is_paid)) < 6:
                is_paid = None
        
        if re.match(
          r'^[\W_]?\d{2}[\W_]?\d{2}[\W_]?(\d{2}|\d{4})[\W_]?$',
          str(is_paid)
          ) or re.match(r'^[\W_]?\d{2}[\W_]?$', str(is_paid)) \
          or (re.match(r'^[\W_]?\d{2}[\W_]?\d{2}[\W_]?$', str(is_paid)) \
          and i+1 < len(lst) and (re.match(r'\D+', str(lst[i])) \
          or re.match(r'^[\W_]?\d{2}[\W_]?(\d{2}|\d{4})[\W_]?$', str(lst[i])) \
          or re.match(r'^[\W_]?(\d{2}|\d{4})[\W_]?$', str(lst[i])))):
            i -= 1
            continue

        if date or donation_type or is_paid:
            df.loc[len(df)] = [date, donation_type, is_paid]
        else:
            i += 1
    
    df['non_empty_count'] = df.apply(lambda x: x.count(), axis=1)
    df = df.sort_values('non_empty_count', ascending=False)
    df = df.drop_duplicates(subset='date', keep='first')
    df = df.drop('non_empty_count', axis=1)
    
    date_copy = df['date'].copy()
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    invalid_rows = df[df['date'].isna()].copy()
    invalid_rows['date'] = date_copy[df['date'].isna()]
    df = df[~df['date'].isna()]
    df = df.sort_values(by='date')
    df['date'] = df['date'].dt.strftime('%d.%m.%Y')
    for idx, row in invalid_rows.iterrows():
        df = pd.concat([df.iloc[:idx], row.to_frame().T, df.iloc[idx:]]).reset_index(drop=True)

    df = df.fillna(value={
        'is_paid': 'Безвозмездно',
        'donation_type': 'Цельная кровь'
        })
        
    return df
