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
            r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*\d{2}[\W_]*(\d{2}|\d{4})[\W_]*$',
            lst[i]
        ) and i+2 < len(lst) and re.match(
            r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]?$',
            lst[i+1]
        ):
            date_parts = re.findall(r'(\d{2})[\W_]*(\d{2})', lst[i])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            if re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]$', lst[i+1]):
                donation_type = re.findall(
                    r'\w{1,2}[\W_]?\w{,2}',
                    lst[i+1]
                )[0].lower()
                is_paid = re.findall(
                    r'[\W_]?\w{,4}[\W_]$',
                    lst[i+1]
                )[0].lower()
                i += 2
            elif re.match(r'^[\W_]?\w{1,4}[\W_]?$', lst[i+2]):
                donation_type = lst[i+1].lower()
                is_paid = lst[i+2].lower()
                i += 3
            else:
                date = None

        elif re.match(r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*$', lst[i]) \
                and i+3 < len(lst) \
                and re.match(r'^[\W_]*\d{2}[\W_]*(\d{2}|\d{4})[\W_]*$', lst[i+1]) \
                and re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]?$', lst[i+2]):
            date_parts = re.findall(
                r'(\d{2})[\W_]*(\d{2})', lst[i]+lst[i+1])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+1])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            if re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]$', lst[i+2]):
                donation_type = re.findall(
                    r'\w{1,2}[\W_]?\w{,2}',
                    lst[i+2]
                )[0].lower()
                is_paid = re.findall(
                    r'[\W_]?\w{,4}[\W_]$',
                    lst[i+2]
                )[0].lower()
                i += 3
            elif re.match(r'^[\W_]?\w{1,4}[\W_]?$', lst[i+3]):
                donation_type = lst[i+2].lower()
                is_paid = lst[i+3].lower()
                i += 4
            else:
                date = None

        elif re.match(r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*\d{2}[\W_]*$', lst[i]) \
                and i+3 < len(lst) \
                and re.match(r'^[\W_]*(\d{2}|\d{4})[\W_]*$', lst[i+1]) \
                and re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]?$', lst[i+2]):
            date_parts = re.findall(r'(\d{2})[\W_]*(\d{2})', lst[i])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+1])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            if re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]$', lst[i+2]):
                donation_type = re.findall(
                    r'\w{1,2}[\W_]?\w{,2}',
                    lst[i+2]
                )[0].lower()
                is_paid = re.findall(
                    r'[\W_]?\w{,4}[\W_]$',
                    lst[i+2]
                )[0].lower()
                i += 3
            elif re.match(r'^[\W_]?\w{1,4}[\W_]?$', lst[i+3]):
                donation_type = lst[i+2].lower()
                is_paid = lst[i+3].lower()
                i += 4
            else:
                date = None

        elif re.match(r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*$', lst[i]) \
                and i+4 < len(lst) \
                and re.match(r'^[\W_]*\d{2}[\W_]*$', lst[i+1]) \
                and re.match(r'^[\W_]*(\d{2}|\d{4})[\W_]*$', lst[i+2]) \
                and re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]?$', lst[i+3]):
            date_parts = re.findall(
                r'(\d{2})[\W_]*(\d{2})', lst[i]+lst[i+1])[0]
            year = '20' + re.findall(r'(\d{2})', lst[i+2])[-1]
            date = f'{date_parts[0]}.{date_parts[1]}.{year}'
            if re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]$', lst[i+3]):
                donation_type = re.findall(
                    r'\w{1,2}[\W_]?\w{,2}',
                    lst[i+3]
                )[0].lower()
                is_paid = re.findall(
                    r'[\W_]?\w{,4}[\W_]$',
                    lst[i+3]
                )[0].lower()
                i += 4
            elif re.match(r'^[\W_]?\w{1,4}[\W_]?$', lst[i+4]):
                donation_type = lst[i+3].lower()
                is_paid = lst[i+4].lower()
                i += 5
            else:
                date = None

        if donation_type:
            if 't' in donation_type:
                donation_type = 'platelets'
            elif 'k' in donation_type \
                    or 'w' in donation_type \
                    or 'p' in donation_type:
                donation_type = 'blood'
            elif 'n' in donation_type \
                    or 'm' in donation_type \
                    or 'v' in donation_type:
                donation_type = 'plasma'
            elif re.search(r'[a-z]', donation_type) \
                    or len(re.findall(r'\d', donation_type)) < 2:
                donation_type = None

        if is_paid:
            if 0 < len(re.findall(r'[a-z0-9]', is_paid)) < 3:
                is_paid = 'free'
            elif 2 < len(re.findall(r'[a-z0-9]', is_paid)) < 5 \
                    and ('n' in is_paid or 't' in is_paid):
                is_paid = 'payed'
            elif len(re.findall(r'[a-z0-9]', is_paid)) > 14 \
                    or len(re.findall(r'[a-z0-9]', is_paid)) == 0 \
                    or len(re.findall(r'\d', is_paid)) < 6:
                is_paid = None

        if (
            re.match(r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*$', str(is_paid)) or
            re.match(
                r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*\d{2}[\W_]*$', str(is_paid))
        ) and i+1 < len(lst) and (
            re.match(r'^[\W_]*\d{2}[\W_]*(\d{2}|\d{4})[\W_]*$', lst[i]) or
            re.match(r'^[\W_]*(\d{2}|\d{4})[\W_]*$', lst[i])
        ):
            i -= 1
            continue

        if (
            re.match(
                r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*\d{2}[\W_]*(\d{2}|\d{4})[\W_]*$',
            str(donation_type)
          ) or
          re.match(
              r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*$',
              str(donation_type)
          ) or
          re.match(
              r'^[^\w\s\(\)]*\d{2}[^\w\s\(\)]*\d{2}[\W_]*$',
            str(donation_type)
          )
        ) and i+1 < len(lst) and (
            re.match(r'^\w{1,2}[\W_]?\w{,2}[\W_]?\w{,4}[\W_]?$', str(is_paid)) or
            re.match(r'^[\W_]*\d{2}[\W_]*(\d{2}|\d{4})[\W_]*$', str(is_paid)) or
            re.match(r'^[\W_]*(\d{2}|\d{4})[\W_]*$', str(is_paid))
        ):
            i -= 2
            continue

        if date:
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
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    for idx, row in invalid_rows.iterrows():
        df = pd.concat([df.iloc[:idx], row.to_frame().T,
                       df.iloc[idx:]]).reset_index(drop=True)

    df = df.fillna(value={
        'is_paid': 'free',
        'donation_type': 'blood'
    })

    df.columns = ['donate_at', 'blood_class', 'payment_type']

    return df
