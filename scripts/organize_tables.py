import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
import re
from scrape import SECScraper
import pandera as pa
from pandera.typing import DataFrame,Series
from pandera import DataFrameSchema

class ScrapeDict:
    def __init__(self, scraper: SECScraper,target_url: str, cutoff_date: str, form_type:str):
        
        self.scraper = scraper
        self.target_url = target_url
        self.cutoff_date = cutoff_date
        self.form_type = form_type
        
        self.soup_dict = self.make_soup_dict()
        
    def make_soup_dict(self):

        scraper = self.scraper
        proxy_soup_dict = scraper.run(self.cutoff_date, self.form_type)

        return proxy_soup_dict

class EBITDA:
    def __init__(self, soup_obj: ScrapeDict, table_identifier: str, table_position0:int, table_position1:int, split_key = '2022-09-28'):

        self.soup_dict = soup_obj.soup_dict
        self.table_identifier = table_identifier
        self.table_position0 = table_position0
        self.table_position1 = table_position1
        self.split_key = split_key

        self.soup_dict_pos0, self.soup_dict_pos1 = self.split_dict_after_key()
        self.full_df, self.mismatches = self.make_clean_table()

    def split_dict_after_key(self):
        original_dict = self.soup_dict
        split_key = self.split_key
        ''' The order of the desired table changes over time so split
        the dict of BS4 html objects into two to process separately.'''
        dict1 = {}
        dict2 = {}
        add_to_second = False

        for key, value in original_dict.items():
            if add_to_second:
                dict2[key] = value
            else:
                dict1[key] = value
                if key == split_key:
                    add_to_second = True

        return dict1, dict2
    
    def make_clean_table(self):
        """Starting from dicts of bs4 objects for each table position, 
        make intermediate FormTables object for each before making
        CleanedEBITDATables object that contains cleaned df of EBITDA for all years
        """

        soup_dict_pos0 = self.soup_dict_pos0
        table_pos0 = self.table_position0

        soup_dict_pos1 = self.soup_dict_pos1
        table_pos1 = self.table_position1

        table_identifier = self.table_identifier

        raw_tables_pos0 = FormTables(soup_dict_pos0, table_identifier, table_pos0).tables_dict
        raw_tables_pos1 = FormTables(soup_dict_pos1, table_identifier, table_pos1).tables_dict

        cleaned_ebitda_obj = CleanedEBITDATables(raw_tables_pos0, raw_tables_pos1)
        ebitda_df = cleaned_ebitda_obj.full_df
        ebitda_df['Amount'] = ebitda_df['Amount'] * 1000
        mismatches = cleaned_ebitda_obj.mismatches

        return ebitda_df, mismatches

class BuyBacks:
    def __init__(self, soup_obj: ScrapeDict, table_identifier: str, table_position:int):

        self.soup_dict = soup_obj.soup_dict
        self.table_identifier = table_identifier
        self.table_position = table_position

        self.raw_tables_dict = self.make_raw_tables()
        self.buyback_df, self.mismatches = self.make_clean_table()

    def make_raw_tables(self):

        soup_dict = self.soup_dict
        table_identifier = self.table_identifier
        table_position = self.table_position
        raw_tables_dict = FormTables(soup_dict, table_identifier, table_position)

        return raw_tables_dict
    
    def make_clean_table(self):

        raw_tables_dict = self.raw_tables_dict
        table_identifier = self.table_identifier
        
        cleaned_tables_obj = CleanedBuybackTables(raw_tables_dict, table_identifier)
        cleaned_df = cleaned_tables_obj.buyback_df
        mismatches = cleaned_tables_obj.mismatches

        cleaned_df['Year'] = pd.to_datetime(cleaned_df['Year'], format='%Y')

        return cleaned_df, mismatches

class RevExp:
    def __init__(self, soup_obj: ScrapeDict, table_identifier: str, table_position:int):

        self.soup_dict = soup_obj.soup_dict
        self.table_identifier = table_identifier
        self.table_position = table_position

        self.raw_tables_dict = self.make_raw_tables()
        self.rev_df, self.exp_df, self.mismatches_rev, self.mismatches_exp = self.make_cleaned_table()

    def make_raw_tables(self):
        soup_dict = self.soup_dict
        table_identifier = self.table_identifier
        table_position = self.table_position
        raw_tables_dict = FormTables(soup_dict, table_identifier, table_position)

        return raw_tables_dict
    def make_cleaned_table(self):
        raw_tables_dict = self.raw_tables_dict
        table_identifier = self.table_identifier
        cleaned_tables_obj = CleanedRevenueExpTables(raw_tables_dict, table_identifier)
        cleaned_tables_obj.combine_dfs()

        rev_df = cleaned_tables_obj.rev_df_full
        exp_df = cleaned_tables_obj.exp_df_full

        mismatches_rev = cleaned_tables_obj.mismatches_rev
        mismatches_exp = cleaned_tables_obj.mismatches_exp

        return rev_df, exp_df, mismatches_rev, mismatches_exp
       
class ExecComp:
    def __init__(self, soup_obj: Dict, table_identifier, table_identifier1):

        self.soup_dict = soup_obj#.soup_dict
        self.table_identifier = table_identifier
        self.table_identifier1 = table_identifier1
        
        
        self.full_df, self.mismatches = self.combine_ec_dfs()

    def check_for_issues(self):

        df = self.full_df
        target_val = 2 #indicates 2 unique vals in cell w/ same name, year
        counts = df.groupby(['Name','Fiscal Year']).nunique()
        locations = counts.stack()[df.stack() == target_val].index.tolist()
        names = [locations[i][0] for i in range(len(locations))]
        years = [locations[i][1] for i in range(len(locations))]
        column_name = [locations[i][2] for i in range(len(locations))]

        #make df 
        mismatch_df = pd.DataFrame({'Name':names,
                          'Year':years,
                          'Column':column_name})
        return mismatch_df

    def combine_ec_dfs(self):

        proxy_soup_dict = self.soup_dict
        #print('orig keys ', proxy_soup_dict.keys())

        #separate dicts
        key_24 = '2024-10-23'

        #for some reason more forms are being picked up now
        # they have been filed in september. want to only keep october forms
        month_keep = '10'

        subset_dict = {key: value for key, value in proxy_soup_dict.items() if key.split('-')[1] == month_keep}
        #print('keys after removign any non october ',subset_dict.keys())

        proxy_24_dict = {key_24: subset_dict[key_24]} if key_24 in subset_dict else {}
        #print('proxy_24_dict keys ', proxy_24_dict.keys())
        assert len(proxy_24_dict.keys()) == 1, 'proxy_24_dict has more than 1 key'

        if key_24 in subset_dict:
            del subset_dict[key_24]

        #proxy_other_dict = {k: v for k, v in proxy_soup_dict.items() if k != key_24}
        assert key_24 not in subset_dict, 'key_24 not in proxy_other_dict'
        #print(proxy_soup_dict.keys())
        base_salary_identifier = self.table_identifier
        base_salary_identifier1 = self.table_identifier1
        
        exec_comp24_obj = FormTablesExecComp24(proxy_24_dict, base_salary_identifier,0)
        ec_24_df = exec_comp24_obj.full_df
        ec_24_mismatches = exec_comp24_obj.mismatches

        #ec_df_24 = FormTablesExecComp24(proxy_24_dict, base_salary_identifier,0).full_df

        exec_comp_pre_obj = FormTablesExecComp(subset_dict, base_salary_identifier1,0)
        ec_pre_df = exec_comp_pre_obj.full_df
        ec_pre_mismatches = exec_comp_pre_obj.mismatches
        if ec_24_mismatches is None and ec_pre_mismatches is None:
            mismatches = None
        else:
            mismatches = pd.concat([ec_24_mismatches, ec_pre_mismatches], ignore_index=True)
        #ec_df_pre = FormTablesExecComp(proxy_other_dict, base_salary_identifier1,0).full_df

        combined_ec_df = pd.concat([ec_24_df, ec_pre_df], ignore_index=True)
        numeric_cols = list(combined_ec_df.select_dtypes(include=['number']).columns)
        combined_ec_df = combined_ec_df.drop(columns = ['orig_file'])
        combined_ec_df = combined_ec_df.drop_duplicates(subset = numeric_cols, keep='first')
        ## TODO fix issue with kirsten lynch 2022 non equity... typo

        #remove empty rows
        #numeric_cols = combined_ec_df.select_dtypes(include=['float']).columns
        final_df = combined_ec_df[~(combined_ec_df[numeric_cols] == 0).all(axis=1)]

        #final_df.loc[:,'orig_file'] = pd.to_datetime(final_df['orig_file'], format = '%Y-%m-%d')
        final_df.loc[:,'Fiscal Year'] = pd.to_datetime(final_df['Fiscal Year'], format = '%Y').dt.year
        
        #final_df = final_df.set_index(['Name','Fiscal Year'])
        #final_df.sort_index(level='Fiscal Year',inplace=True, ascending=False)
        return final_df, mismatches

def check_for_duplicates_errors_exec_comp(df):
        '''This function should be run after combining multiple years of data.
    Because each financial form has 3 years, there are many duplicates in
    the combined dataframe. This function checks for duplicates and checks
    that all duplicates have identical values'''

        df = df.drop(columns = ['orig_file'])
        key_columns = ['Name', 'Fiscal Year'] # these are like header cols
        grouped = df.groupby(key_columns).nunique()
        #print('grouped: ', grouped)
        inconsistent_groups = grouped[(grouped > 1).any(axis=1)]
        if len(inconsistent_groups) == 0:
            print('No mismatches, looks all good')
            return None
            #assert len(inconsistent_groups) == 0, 'There are issues in the df. not all duplicates are identical'
        else:
            print('Mismatch found')
            target_val = 2 #indicates 2 unique vals in cell w/ same name, year
            counts = df.groupby(['Name','Fiscal Year']).nunique()

            stacked = counts.stack()

            bool_series = stacked == target_val

            aligned_bool_series = bool_series[bool_series]

            locations = aligned_bool_series.index.tolist()
            #locations = counts.stack()[df.stack() == target_val].index.tolist()
            names = [locations[i][0] for i in range(len(locations))]
            years = [locations[i][1] for i in range(len(locations))]
            column_name = [locations[i][2] for i in range(len(locations))]

            #make df 
            mismatch_df = pd.DataFrame({'Name':names,
                            'Year':years,
                            'Column':column_name})
            return mismatch_df
            
            
        
        
        #return inconsistent_groups 
class FormTablesExecComp24:
    def __init__(self, soup_dict, table_identifier, table_position):
        self.soup_dict = soup_dict
        self.table_identifier = table_identifier
        self.table_position = table_position
        self.tables_dict = self.find_tables_runner()
        self.processed_df = self.process_tables_runner()
        self.full_df, self.mismatches = self.final_clean()
    
    def find_tables_span(self, soup_obj, year: str) -> Optional[pd.DataFrame]:
        #print('in find tables')
        try:
            span = soup_obj.find('span', string=self.table_identifier)
            if span:
                #print('found key')
                table = span.find_parent('table')
                if table:
                    rows = table.find_all('tr')
                    
                    # Initialize a list for the table structure
                    table_data = []
                    max_columns = 0

                    # First pass: determine the table size
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        max_columns = max(max_columns, sum(int(cell.get('colspan', 1)) for cell in cells))
                    
                    # Create a matrix to hold cell data
                    table_matrix = []
                    for _ in range(len(rows)):
                        table_matrix.append([None] * max_columns)
                    
                    # Second pass: populate the table matrix
                    for row_idx, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        col_idx = 0

                        for cell in cells:
                            # Skip filled cells
                            while col_idx < max_columns and table_matrix[row_idx][col_idx] is not None:
                                col_idx += 1

                            # Extract cell text
                            cell_text = cell.get_text(strip=True)
                            colspan = int(cell.get('colspan', 1))
                            rowspan = int(cell.get('rowspan', 1))

                            # Fill cell data in the matrix
                            for r in range(row_idx, row_idx + rowspan):
                                for c in range(col_idx, col_idx + colspan):
                                    table_matrix[r][c] = cell_text

                            col_idx += colspan  # Move to next cell after colspan

                    # Convert matrix to DataFrame
                    df = pd.DataFrame(table_matrix)
                    return df

        except:
            print('no dice')

    def find_tables_runner(self) -> Dict[str, pd.DataFrame]:
        key_ls, df_ls = [],[]

        for key in self.soup_dict.keys():
            #print(f'Processing year: {key}')
            try:
                soup = self.soup_dict[key]
                df = self.find_tables_span(soup, key)
                df_ls.append(df)
                key_ls.append(key)
        
                #if df is not None:
                #    df_dict[key] = df
                #else:
                #    pass
                    #create empty df w same structure
                    #df_dict[key] = pd.DataFrame(columns = ['Year',self.table_identifier])
            except Exception as e:
                self.logger.error(f'Error processing tables for year {key}: {str(e)}')
                self.missing_data_years.append(key)
                #create empty df w same structure
                #df_dict[key] = pd.DataFrame(columns = ['Year',self.table_identifier])
        #if self.missing_data_years:
            #self.logger.warning(f'Missing data for years: {', '.join(self.missing_data_years)}')
        #    self.logger.warning(f"Missing data for years: {', '.join(self.missing_data_years)}")
        df_dict = dict(zip(key_ls, df_ls))
        return df_dict 

    def process_tables(self, df):

        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        numeric_cols = df.columns.to_list()[2:]
        numeric_cols = [i.split('($)')[0] for i in numeric_cols]

        new_cols = ['Name and Principal Position', 'Fiscal Year'] + numeric_cols
        df.columns = new_cols
        for col in numeric_cols:
            if df[col].dtype == "object":  # Ensure the column is of string type
                df[col] = (
                    df[col]
                    .replace(",", "", regex=True)  # Remove commas
                    .replace("—", "0", regex=True)  # Replace '—' with 0
                    .replace("", "0", regex=True)  # Replace empty strings with 0
                    .astype(float)  # Convert to float first
                    .fillna(0)  # Replace NaNs with 0
                    .astype(int)  # Convert to integers
            )
        df = df.iloc[1:,:-1]
        return df
    
    def process_tables_runner(self):
        key_ls, df_ls = [],[]
        
        for key in self.tables_dict.keys():
            raw_df = self.tables_dict[key]
            clean_df = self.process_tables(raw_df)
            clean_df['orig_file'] = key
            df_ls.append(clean_df)
            key_ls.append(key)
        df_dict = dict(zip(key_ls, df_ls))
        full_df = pd.concat(df_dict.values(), axis=0, ignore_index=True)
        return full_df 

    def final_clean(self):

        exec_names = ['Kirsten A. Lynch', 'Robert A. Katz','Angela A. Korch','William C. Rock',
              'David T. Shapiro','Gregory J. Sullivan']
        names_ls, pos_ls, year_ls, salary_ls, bonus_ls, stock_ls, option_ls, nonequity_ls, other_ls, total_ls, orig_file_ls = [],[],[],[],[],[],[],[],[],[],[]
        df = self.processed_df
        for _,row in df.iterrows():
            year = int(row['Fiscal Year'])
            year_ls.append(year)
            salary = float(row['Salary '])
            salary_ls.append(salary)
            
            stock = float(row['Stock Awards'])
            stock_ls.append(stock)
            option = float(row['Option/Share Appreciation Right Awards'])
            option_ls.append(option)
            nonequity = float(row['Non-Equity Incentive Plan Compensation'])
            nonequity_ls.append(nonequity)
            other = float(row['All Other Compensation'])
            other_ls.append(other)
            total = float(row['Total'])
            total_ls.append(total)
            orig_file = row['orig_file']
            orig_file_ls.append(orig_file)
            #print('year ', year)
            row_str = row['Name and Principal Position']
            target_name = None
            #Now find correct name to match row
            for name in exec_names:
                if row_str.startswith(name):
                    target_name = name
                    idx_start = row_str.find(name)
                    idx_end = idx_start + len(name)
                    row_name = row_str[idx_start:idx_end]
                    #print('row name ', row_name)
                    #print('total ', row['Total'])
                    names_ls.append(row_name)
                    row_pos =  row_str[idx_end:]
                    pos_ls.append(row_pos)
            
        new_df= pd.DataFrame({'Name':names_ls, 'Position':pos_ls,
                            'Fiscal Year':year_ls, 'Salary':salary_ls,
                             'StockAwards':stock_ls,
                            'Option/ShareAppreciationRight Awards':option_ls,
                            'Non-EquityIncentive PlanCompensation':nonequity_ls,
                            'All OtherCompensation':other_ls, 'Total':total_ls,
                            'orig_file':orig_file_ls
                            })
        mismatches = check_for_duplicates_errors_exec_comp(new_df)

        new_df = new_df.drop_duplicates(keep='first')
        return new_df, mismatches

class FormTablesExecComp:
    def __init__(self, soup_dict, table_identifier, table_position):
        self.soup_dict = soup_dict
        self.table_identifier = table_identifier
        self.table_position = table_position
        
        self.a_df, self.b_df = self.process_tables_runner() 
        self.update_df_a()
        self.combined_df = pd.concat([self.a_df, self.b_df], ignore_index=True)
        self.full_df, self.mismatches = self.final_clean()
    
    def check_for_duplicates_errors(self):
        '''This function should be run after combining multiple years of data.
    Because each financial form has 3 years, there are many duplicates in
    the combined dataframe. This function checks for duplicates and checks
    that all duplicates have identical values'''

        df = self.full_df
        df_sub = df.drop(columns=['orig_file']) #this is added after for tracking purposes (won't have duplicates)
        key_columns = ['Name', 'Position', 'Fiscal Year'] # these are like header cols
        grouped = df_sub.groupby(key_columns).nunique()
        inconsistent_groups = grouped[(grouped > 1).any(axis=1)]
        assert len(inconsistent_groups) == 0, 'There are issues in the df. not all duplicates are identical'
        #return inconsistent_groups

    def final_clean(self):
        df = self.combined_df
        mismatches = check_for_duplicates_errors_exec_comp(df)

        df = df.drop_duplicates(keep='first')
        return df, mismatches
    def update_df_a(self):

        df = self.a_df
        #exec_names = tuple(self.b_df['Name'].unique().tolist())
        exec_names = ['Kirsten A. Lynch', 'Robert A. Katz', 'Angela A. Korch',
                      'Michael Z. Barkin', 'James. C. O’Donnell' ,
                      'David T. Shapiro', 'Ryan Bennett']
        names_ls, pos_ls = [],[]

        for _,row in df.iterrows():

            row_str = row['Name and Principal Position']
            target_name = None
            #Now find correct name to match row
            for name in exec_names:
                if row_str.startswith(name):
                    target_name = name
                    idx_start = row_str.find(name)
                    idx_end = idx_start + len(name)
                    row_name = row_str[idx_start:idx_end]
                    names_ls.append(row_name)
                    row_pos =  row_str[idx_end:]
                    pos_ls.append(row_pos)
        df[['Name','Position']] = pd.DataFrame({'Name':names_ls,
                                                'Position':pos_ls})
        cols_to_move = ['Name', 'Position']
        new_order = cols_to_move + [col for col in df.columns if col not in cols_to_move]
        df = df[new_order]
        df = df.drop(columns=['Name and Principal Position'])
        #Update a_df attr
        self.a_df = df
            

    def process_tables_runner(self):
        a_keys_ls, b_keys_ls, a_dfs_ls, b_dfs_ls = [],[],[],[]
        for key in self.soup_dict.keys():

            if key in ['2014-10-23','2015-10-22']:
                pass
            else:
                #print(key)
                soup_obj = self.soup_dict[key]
                table_identifier = self.table_identifier
                key = key
                year = key[:4]
                    
                raw_df = self.identify_table(soup_obj, table_identifier)
                if year in ['2022','2023']:
                    clean_df = self.process_a(raw_df)
                    clean_df['orig_file'] = key
                    a_keys_ls.append(key)
                    a_dfs_ls.append(clean_df)

                else:
                    clean_df = self.process_b(raw_df)
                    clean_df['orig_file'] = key
                    b_keys_ls.append(key)
                    b_dfs_ls.append(clean_df)
        a_data_dict = dict(zip(a_keys_ls, a_dfs_ls))
        b_data_dict = dict(zip(b_keys_ls, b_dfs_ls))

        a_data_df = pd.concat(a_data_dict.values(), ignore_index=True)
        b_data_df = pd.concat(b_data_dict.values(), ignore_index=True)   
            
       
        return a_data_df, b_data_df

    def identify_table(self, soup_obj, table_identifier):
        tables = soup_obj.find_all('table')
        for i, table in enumerate(tables):
            for row in table.find_all('tr'):
                if table_identifier in row.get_text():
                    #print('i ', i)
                    #print('found table id')
                    table_idx = i
                    
        table = tables[table_idx]

        if table:
            rows = table.find_all('tr')
            
            # Initialize a list for the table structure
            table_data = []
            max_columns = 0

            # First pass: determine the table size
            for row in rows:
                cells = row.find_all(['td', 'th'])
                max_columns = max(max_columns, sum(int(cell.get('colspan', 1)) for cell in cells))
            
            # Create a matrix to hold cell data
            table_matrix = []
            for _ in range(len(rows)):
                table_matrix.append([None] * max_columns)
            
            # Second pass: populate the table matrix
            for row_idx, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                col_idx = 0

                for cell in cells:
                    # Skip filled cells
                    while col_idx < max_columns and table_matrix[row_idx][col_idx] is not None:
                        col_idx += 1

                    # Extract cell text
                    cell_text = cell.get_text(strip=True)
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))

                    # Fill cell data in the matrix
                    for r in range(row_idx, row_idx + rowspan):
                        for c in range(col_idx, col_idx + colspan):
                            table_matrix[r][c] = cell_text

                    col_idx += colspan  # Move to next cell after colspan

            # Convert matrix to DataFrame
            df = pd.DataFrame(table_matrix)

            # String to search for
            search_string = 'Name and Principal Position'

            # Find rows where any column contains the search string
            matching_rows = df.map(lambda x: search_string in str(x)).any(axis=1)
            matching_row_indices = matching_rows[matching_rows].index
            idx = matching_row_indices[0]
            #print(idx)

            df.columns = df.iloc[idx]

            #df.columns = df.iloc[0]
            #print(df.columns)
            if idx == 0:
                idx = 1
            df = df.iloc[idx:].reset_index(drop=True)
            if df.iloc[0,0] == 'Name and Principal Position':
                df = df.iloc[1:]
            df_cleaned = df.dropna(axis=1, how='all')

            return df_cleaned

    def process_b(self,df):#,columns_to_keep = cols_to_keep):
        #print('in b')
        df = df.loc[:, ~df.columns.duplicated()]
        #remove any hidden chars
        df.columns = df.columns.str.strip().str.replace(r'\u200b', '', regex=True)

        df = df.rename({'FiscalYear':'Fiscal Year'},axis=1)
        
        if 'FiscalYear' in df.columns:
            print('wtf')
        # Remove zero-width spaces and empty strings from column names
        df.columns = df.columns.str.replace('​', '').str.strip()
        df = df.replace('—', 0)
        df = df.replace(",", "")
        df = df.replace("",0)
        df['Fiscal Year'] = df['Fiscal Year'].astype(int)

        df.columns = [col.split('($)')[0].strip() if '($)' in col else col for col in df.columns]
        
        #fix spelling issue in other column
        if 'All OtherCompen-sation' in df.columns:
            df = df.rename(columns={'All OtherCompen-sation': 'All OtherCompensation'})
        if 'Option/ShareAppreciationRightAwards' in df.columns:
            df = df.rename(columns={'Option/ShareAppreciationRightAwards': 'Option/ShareAppreciationRight Awards'})
        def replace_parentheses(value):
            value = re.sub(r'\(.*\)', '', str(value))
            value = value.replace(',','')
            return value

        # Apply regex replacement across all columns to remove anything in parentheses
        df = df.map(replace_parentheses)
        #cleaned_numeric_cols = df_cleaned.columns[2:]
        names, positions = [], []
        other_columns = {col: [] for col in df.columns if col != 'Name and Principal Position'}

        #this next section is to separate names and positions and 
        # to handle different group sizes
        def is_name(value):
            return bool(re.match(r'\b[A-Za-z]+\s[A-Za-z]\.\s[A-Za-z]+\b', value))
        def group_rows_by_year(df):
            grouped_data = []
            group = []
            previous_year = None
            
            for i, row in df.iterrows():
                current_year = row['Fiscal Year']
                
                # Check if we're continuing the descending order or starting a new group
                if previous_year is None or current_year < previous_year:
                    group.append(row)
                else:
                    # If the year starts ascending, save the previous group and start a new one
                    if len(group) > 1:
                        grouped_data.append(group)  # Append the completed group
                    group = [row]  # Start a new group with the current row
                
                previous_year = current_year  # Update the previous year
            
            # Append the last group (if exists)
            if len(group) > 1:
                grouped_data.append(group)
            
            return grouped_data
        
        grouped_rows = group_rows_by_year(df)
        for group in grouped_rows:
            name = group[0]['Name and Principal Position'] if is_name(group[0]['Name and Principal Position']) else None
            position = group[1]['Name and Principal Position'] if len(group) > 1 else None

            names.extend([name] * len(group))
            positions.extend([position] * len(group))

            for col in other_columns.keys():
                other_columns[col].extend([row[col] for row in group])
        df_final = pd.DataFrame({'Name': names, 'Position': positions, **other_columns})
        
        
        

        col_drop1 = 'Change inPension Value and Non-qualified DeferredCompensationEarnings'
        col_drop2 = 'Change inPensionValue andNonqualifiedDeferredCompensationEarnings'
        if col_drop1 in df_final.columns:
            df_final = df_final.drop(col_drop1, axis=1)
        elif col_drop2 in df_final.columns:
            df_final = df_final.drop(col_drop2, axis=1)
            
        #remove fiscalyear col (not sure how its still getting in)
        if 'FiscalYear' in df_final.columns:
            df_final = df_final.drop('FiscalYear', axis=1)
        

        df_final = df_final.loc[:, df_final.columns != ''] 
        cols = df_final.columns.to_list()
        #print(cols)
        dtype_dict = ({
            'Name':'str','Position':'str',
            'Fiscal Year':'int','Salary':'float',
            'Bonus':'float','StockAwards':'float',
            'Option/ShareAppreciationRight Awards':'float',
            'Non-EquityIncentive PlanCompensation':'float',
            'All OtherCompensation':'float','Total':'float'})
            
        df_final = df_final.astype(dtype_dict)
        keep_idxs = [0,1,2,3,5,6,7,8,9] #Name, Position, Fiscal Year, Salary, Total
        cols_to_keep = [cols[i] for i in keep_idxs]
        df_final = df_final[cols_to_keep]

        return df_final

    def process_a(self,df):#,columns_to_keep = cols_to_keep):
        #print('in a')
        df = df.rename({'FiscalYear':'Fiscal Year'},axis=1)
        # Remove zero-width spaces and empty strings from column names
        df.columns = df.columns.str.replace('​', '').str.strip()
        df = df.replace('—', 0)
        df = df.replace(",", "")
        df = df.replace("",0)


        #print(df.columns)

        # If necessary, remove columns that are completely empty or contain only whitespace
        df_cleaned = df.loc[:, df.columns.str.strip() != '']  # Keep columns that aren't just empty strings or spaces

        numeric_cols = df_cleaned.columns.to_list()[2:]
        numeric_cols = [i.split('($)')[0] for i in numeric_cols]
        new_cols = new_cols = ['Name and Principal Position', 'Fiscal Year'] + numeric_cols
        df_cleaned.columns = new_cols
        def replace_parentheses(value):
            return re.sub(r'\(.*\)', '', str(value))

        # Apply regex replacement across all columns to remove anything in parentheses
        df_cleaned = df_cleaned.map(replace_parentheses)
        cleaned_numeric_cols = df_cleaned.columns[2:]

        for col in cleaned_numeric_cols:
            # Check if the column is a Series and its dtype is 'object' (string)
            if isinstance(df_cleaned[col], pd.Series):  # Ensure it's a Series
                if df_cleaned[col].dtype == "object":  # Ensure it's of string type
                    # Remove commas from string-type columns
                    df_cleaned[col] = df_cleaned[col].replace(",", "", regex=True)  # Remove commas
                                                    
                    
                    # Additional step: You can convert to numeric if required
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
        
        df_final = df_cleaned.loc[:, ~df_cleaned.columns.duplicated()]
        df_final = df_final.loc[:, df_final.columns != ''] 
        #print(df_final.columns)
        cols = df_final.columns.to_list()
        dtype_dict = ({
            'Name and Principal Position':'str',
            'Fiscal Year':'int','Salary':'float',
            'Bonus':'float', 'StockAwards':'float', 
            'Option/ShareAppreciationRight Awards':'float',
            'Non-EquityIncentive PlanCompensation':'float',
            'All OtherCompensation':'float',
             'Total':'float'#,cols[9]:'float'
        })
        df_final = df_final.astype(dtype_dict)
        keep_idxs = [0,1,2,4,5,6,7,8]
        cols_to_keep = [cols[i] for i in keep_idxs]
        df_final = df_final[cols_to_keep]

        # Return the cleaned DataFrame
        return df_final
    def new_process(self, df):

        #ultimately we only want these cols
        desired_columns = ['Name and Principal Position', 'Fiscal Year', 'Salary', 
                           'Bonus', 'All Other Compensation','Total']
        #print('in df process')
        df = df.copy()
        #remoev extra columns
        df = df.loc[:,~df.columns.duplicated()]
        # Drop columns with empty names
        result_df = df.loc[:, df.columns != '']  # Keep only columns with non-empty names
    
        #df.columns = df.columns.str.replace('​', '').str.strip()

        #fix fiscal year col name
        df = df.rename({'FiscalYear':'Fiscal Year'},axis=1)

        #reformat names, positions
        names, positions = [], []
        other_columns = {col: [] for col in df.columns if col != 'Name and Principal Position'}
        def is_name(value):
            return bool(re.match(r'\b[A-Za-z]+\s[A-Za-z]\.\s[A-Za-z]+\b', value))
        
        
        for i in range(0, len(df), 3):
            #print('in col rearrange')
            if i+2 < len(df):
                # Extract name from the first row of the person (based on the pattern)
                name = df.iloc[i]['Name and Principal Position'] if is_name(df.iloc[i]['Name and Principal Position']) else None
                #print(name)
                position = df.iloc[i+1]['Name and Principal Position']
                #print(position)
                # Fill the 'Name' column for the three rows (name in the first row, empty for others)
                names.extend([name, name, name])
                #print('re-org')
                # Fill the 'Position' column for the three rows
                positions.extend([position, position, position])
                for col in other_columns.keys():
                    other_columns[col].extend([df.iloc[i][col], df.iloc[i+1][col], df.iloc[i+2][col]])

        # Create a new DataFrame with 'Name', 'Position', and other columns
        result_df = pd.DataFrame({'Name': names, 'Position': positions, **other_columns})

        result_df.columns = (
        result_df.columns
        .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace to single spaces
        .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters except alphanumerics and spaces
        .str.strip()  # Remove leading and trailing spaces
                    )

        
        #remove duplicate columns
        # Remove zero-width spaces and empty strings from column names
        #df.columns = df.columns.str.replace('​', '').str.strip()
        result_df = result_df.replace('—', 0)
        result_df = result_df.replace("",0)
        result_df = result_df.replace(",", "")
       # result_df.columns = result_df.columns.astype(str)
        result_df.columns = [re.sub(r'\(\$\).*', '', col).strip() for col in result_df.columns]
        #df.columns = [col.split('($)')[0].strip() if '($)' in col else col for col in df.columns]

        #numeric_cols = df.columns.to_list()[2:]
        #numeric_cols = [i.split('($)')[0] for i in numeric_cols]
        #new_cols = new_cols = ['Name and Principal Position', 'Fiscal Year'] + numeric_cols
        #df.columns = new_cols
        def replace_parentheses(value):
            return re.sub(r'\(.*\)', '', str(value))

        # Apply regex replacement across all columns to remove anything in parentheses
        result_df = result_df.map(replace_parentheses)
        col_name = 'Total'
        col_index = result_df.columns.get_loc(col_name)
        result_df = result_df.iloc[:, :col_index+1]
        #print(result_df.columns)
        result_df = result_df[[desired_columns]]
        #print(result_df.columns)
        #assert result_df.columns.to_list() == desired_columns, 'Column names do not match desired columns'


        return result_df
    def old_process(self,df):#,columns_to_keep = cols_to_keep):
        desired_columns = ['Name and Principal Position', 'Fiscal Year', 'Salary',
                            'Bonus', 'All Other Compensation','Total']

        #print(df.columns)
        df = df.rename({'FiscalYear':'Fiscal Year'},axis=1)
        # Remove zero-width spaces and empty strings from column names
        df.columns = df.columns.str.replace('​', '').str.strip()
        df = df.replace('—', 0)
        df = df.replace(",", "")
        df = df.replace("",0)

        df_cleaned = df.loc[:, df.columns.str.strip() != '']  # Keep columns that aren't just empty strings or spaces

        numeric_cols = df_cleaned.columns.to_list()[2:]
        numeric_cols = [i.split('($)')[0] for i in numeric_cols]
        new_cols = new_cols = ['Name and Principal Position', 'Fiscal Year'] + numeric_cols
        df_cleaned.columns = new_cols
        def replace_parentheses(value):
            return re.sub(r'\(.*\)', '', str(value))

        # Apply regex replacement across all columns to remove anything in parentheses
        df_cleaned = df_cleaned.map(replace_parentheses)
        cleaned_numeric_cols = df_cleaned.columns[2:]

        #if columns_to_keep:
        #    # Ensure only columns that exist in the DataFrame are kept
        #    columns_to_keep = list(set(columns_to_keep) & set(df_cleaned.columns))
        #    df_cleaned = df_cleaned[columns_to_keep]

        #df_cleaned = df_cleaned[cols_to_keep]
        for col in cleaned_numeric_cols:
            # Check if the column is a Series and its dtype is 'object' (string)
            if isinstance(df_cleaned[col], pd.Series):  # Ensure it's a Series
                if df_cleaned[col].dtype == "object":  # Ensure it's of string type
                    # Remove commas from string-type columns
                    df_cleaned[col] = df_cleaned[col].replace(",", "", regex=True)  # Remove commas
                                                    
                    
                    # Additional step: You can convert to numeric if required
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
        
        df_final = df_cleaned.loc[:, ~df_cleaned.columns.duplicated()]
        col_name = 'Total'
        col_index = df_final.columns.get_loc(col_name)
        df_final = df_final.iloc[:, :col_index+1]
        df_final = df_final[[desired_columns]]
        #print(df_final.columns)
        #assert df_final.columns.to_list() == desired_columns, 'Column names do not match desired columns'


        # Return the cleaned DataFrame
        return df_final
       
class FormTables:
    def __init__(self, soup_dict, table_identifier, table_position):
        self.soup_dict = soup_dict
        self.table_identifier = table_identifier
        self.table_position = table_position
        self.missing_data_years = []

        # Set up logging - fixed implementation
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

        #print(f'Table position: {self.table_position}')
        self.tables_dict = self.find_tables_runner() #dict of dfs for each year


    def table_contains_row_name(self, table, row_name:str) -> bool:
        for row in table.find_all('tr'):
            if row_name in row.get_text():
                return True
        return False
    
    def extract_table_data(self, table) -> List[List[str]]:
        try:
            rows = []
            for row in table.find_all('tr'):
                cols = row.find_all(['td', 'th'])
                cols = [col.text.strip() for col in cols]
                rows.append(cols)
            return rows
        except AttributeError as e:
            self.logger.error(f'Error extracting table data: {e}')
            raise ValueError('Invalid table structure') from e
        
    def find_tables_span(self, soup_obj, year: str) -> Optional[pd.DataFrame]:
        try:
            span = soup_obj.find('span', string=self.table_identifier)
            if span:
                #print('found key')
                table = span.find_parent('table')
                if table:
                    rows = table.find_all('tr')
                    
                    # Initialize a list for the table structure
                    table_data = []
                    max_columns = 0

                    # First pass: determine the table size
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        max_columns = max(max_columns, sum(int(cell.get('colspan', 1)) for cell in cells))
                    
                    # Create a matrix to hold cell data
                    table_matrix = []
                    for _ in range(len(rows)):
                        table_matrix.append([None] * max_columns)
                    
                    # Second pass: populate the table matrix
                    for row_idx, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        col_idx = 0

                        for cell in cells:
                            # Skip filled cells
                            while col_idx < max_columns and table_matrix[row_idx][col_idx] is not None:
                                col_idx += 1

                            # Extract cell text
                            cell_text = cell.get_text(strip=True)
                            colspan = int(cell.get('colspan', 1))
                            rowspan = int(cell.get('rowspan', 1))

                            # Fill cell data in the matrix
                            for r in range(row_idx, row_idx + rowspan):
                                for c in range(col_idx, col_idx + colspan):
                                    table_matrix[r][c] = cell_text

                            col_idx += colspan  # Move to next cell after colspan

                    # Convert matrix to DataFrame
                    df = pd.DataFrame(table_matrix)
                    return df

        except:
            print('no dice')
            #tables = soup_obj.find_all('table')
            #print('Table identifier: ', self.table_identifier)
            #target_row_name = self.table_identifier #"Mountain net revenue"
            #df_ls, table_id_ls = [], []

            #matching_tables = False
            #for i, table in enumerate(tables):
            #    if self.table_contains_row_name(table, target_row_name):
            #        print('found matching string')
            #        matching_tables = True
            #        table_data = self.extract_table_data(table)
            #        print(len(table_data))

                    #handle empty table data
                    #if not table_data:
                    #    self.logger.warning(f'Empty table data for table {year}')
                    #    continue

            #        toss = table_data[0]
            #        header = table_data[1:]
            #        data = table_data[2:]
                    #print('header: ', header)
                    #data = table_data[1:]
                    #print('data: ', data)

                    #validate data
                    #if not data:
                    #    self.logger.warning(f'No data rows found in table for yea {year}')
                    #    continue

                    #Adjust header length
            #        max_columns = max(len(row) for row in data)
            #        if len(header) < max_columns:
            #            header.extend([''] * (max_columns - len(header)))
            #        elif len(header) > max_columns:
            #            header = header[:max_columns]
            #        df = pd.DataFrame(data, columns=header)
            #        print(df)
            #        df_ls.append(df)
            #        table_id = f'table {i}'
            #        table_id_ls.append(table_id)
            
            #if not matching_tables:
            #    self.logger.warning(f'No tables found containing {target_row_name} for year {year}')
            #    self.missing_data_years.append(year)
            #    return None
            #if not table_id_ls:
            #    self.logger.warning(f'No valid tables processed for year {year}')
            #    self.missing_data_years.append(year)
            #    return None
            #try:
            #    tables_dict = dict(zip(table_id_ls, df_ls))
            #    print('here ', table_id_ls[self.table_position])
            #    key0 = table_id_ls[self.table_position] #0
            #    df_table = tables_dict[key0]
            #    return df_table 
            #except IndexError:
            #    self.logger.error(f'IndexError: Table position {self.table_position} is out of range for year {year}')
            #    self.missing_data_years.append(year)
            #    return None
        #except Exception as e:
        #    self.logger.error(f'Error processing tables for year {year}: {str(e)}')
        #    self.missing_data_years.append(year)
        #    return None
    def find_tables(self, soup_obj, year):

        tables = soup_obj.find_all('table')
        #print('Table identifier: ', self.table_identifier)
        target_row_name = self.table_identifier #"Mountain net revenue"
        df_ls, table_id_ls = [], []

        matching_tables = False
        for i, table in enumerate(tables):
            if self.table_contains_row_name(table, target_row_name):
                #print('found matching string')
                matching_tables = True
                table_data = self.extract_table_data(table)
                #print(len(table_data))

                #handle empty table data
                #if not table_data:
                #    self.logger.warning(f'Empty table data for table {year}')
                #    continue

                header = table_data[0]
                data = table_data[1:]
                #data = table_data[2:]
                #print('header: ', header)
                #data = table_data[1:]
                #print('data: ', data)

                #validate data
                #if not data:
                #    self.logger.warning(f'No data rows found in table for yea {year}')
                #    continue

                #Adjust header length
                max_columns = max(len(row) for row in data)
                if len(header) < max_columns:
                    header.extend([''] * (max_columns - len(header)))
                elif len(header) > max_columns:
                    header = header[:max_columns]
                df = pd.DataFrame(data, columns=header)
                #print(df)
                df_ls.append(df)
                table_id = f'table {i}'
                table_id_ls.append(table_id)
        
        tables_dict = dict(zip(table_id_ls, df_ls))
        #print('here ', table_id_ls[self.table_position])
        key0 = table_id_ls[self.table_position] #0
        df_table = tables_dict[key0]
        return df_table 

    def find_tables_runner(self) -> Dict[str, pd.DataFrame]:
        key_ls, df_ls = [],[]

        for key in self.soup_dict.keys():
            #print(f'Processing year: {key}')
            try:
                soup = self.soup_dict[key]
                #print(key)
                df = self.find_tables(soup, key)
                #print(df.columns)
                df_ls.append(df)
                key_ls.append(key)
        
                #if df is not None:
                #    df_dict[key] = df
                #else:
                #    pass
                    #create empty df w same structure
                    #df_dict[key] = pd.DataFrame(columns = ['Year',self.table_identifier])
            except Exception as e:
                self.logger.error(f'Error processing tables for year {key}: {str(e)}')
                self.missing_data_years.append(key)
                #create empty df w same structure
                #df_dict[key] = pd.DataFrame(columns = ['Year',self.table_identifier])
        #if self.missing_data_years:
            #self.logger.warning(f'Missing data for years: {', '.join(self.missing_data_years)}')
        #    self.logger.warning(f"Missing data for years: {', '.join(self.missing_data_years)}")
        df_dict = dict(zip(key_ls, df_ls))
        return df_dict
    
        #df_ls, key_ls = [],[]

        #for key in self.soup_dict.keys():
        #    print(key)
        #    soup = self.soup_dict[key]
        #    df = self.find_tables(soup)
        #    df_ls.append(df)
        #    key_ls.append(key)

        #tables_dict = dict(zip(key_ls, df_ls))
        #return tables_dict

def check_for_duplicates_errors_ebitda(df):
        '''This function should be run after combining multiple years of data.
    Because each financial form has 3 years, there are many duplicates in
    the combined dataframe. This function checks for duplicates and checks
    that all duplicates have identical values'''

        #df = df.drop(columns = ['orig_file'])
        key_columns = ['Item', 'Year'] # these are like header cols
        grouped = df.groupby(key_columns).nunique()
        inconsistent_groups = grouped[(grouped > 1).any(axis=1)]
        if len(inconsistent_groups) == 0:
            print('No mismatches, looks all good')
            return None
        else:
            print('Mismatch found')
            target_val = 2 #indicates 2 unique vals in cell w/ same name, year
            counts = df.groupby(['Item','Year']).nunique()

            stacked = counts.stack()

            bool_series = stacked == target_val

            aligned_bool_series = bool_series[bool_series]

            locations = aligned_bool_series.index.tolist()
            
            item = [locations[i][0] for i in range(len(locations))]
            year = [locations[i][1] for i in range(len(locations))]
            amount = [locations[i][2] for i in range(len(locations))]

            df_mismatches = pd.DataFrame({'Item': item, 'Year': year, 'Amount': amount})
            return df_mismatches

class CleanedEBITDATables:
    def __init__(self, raw_tables_recent, raw_tables_old):
        self.raw_tables_recent = raw_tables_recent
        self.raw_tables_old = raw_tables_old
        self.full_df, self.mismatches = self.clean_ebitda_tables_driver()
    
    def clean_ebitda_tables_driver(self):
        recent_dict = self.raw_tables_recent
        old_dict = self.raw_tables_old
        
        recent_key_ls, recent_table_ls = [],[]
        for key in recent_dict.keys():

            raw_df = recent_dict[key]
            clean_df = self.clean_ebita_tables(raw_df)

            recent_table_ls.append(clean_df)
            recent_key_ls.append(key)

        recent_dict = dict(zip(recent_key_ls, recent_table_ls))

        old_key_ls, old_table_ls = [],[]
        for key in old_dict.keys():
            raw_df = old_dict[key]
            clean_df = self.clean_ebita_tables(raw_df)

            old_table_ls.append(clean_df)
            old_key_ls.append(key)
        old_dict = dict(zip(old_key_ls, old_table_ls))

        full_dict = {**recent_dict, **old_dict}
        full_df = pd.concat(full_dict.values(), ignore_index=True)

        #check for mismatches
        mismatches = check_for_duplicates_errors_ebitda(full_df)
        print(type(mismatches))
        print(mismatches)
        full_df.drop_duplicates(ignore_index=True, inplace=True)

        return full_df, mismatches

    def clean_ebita_tables(self, df):
        df = df.copy()
        df.columns = [f"Column{i}" for i in range(1, len(df.columns) + 1)]
        #print('df cols: ', df.columns)
        df['Column2'] = df['Column2'].str.normalize('NFKD').str.strip()
        match_index = df[df.apply(lambda row: row.astype(str).str.contains('Year Ended July 31,', case=False, na=False).any(), axis=1)].index

        # If a match is found, subset the DataFrame to rows after the match
        if not match_index.empty:
            df = df.loc[match_index[0] + 1:]  # Select rows after the first match
        #df = df[~df['Column2'].str.contains('Year ended July 31,')]
        df.iloc[0,0] = 'Year'
        df.set_index('Column1',inplace=True)
        df_sub = df.loc[['Year','Mountain Reported EBITDA']]
        

        def convert_value_to_float(cell,n):
            if isinstance(cell, str) and len(cell) == n:
                return float(cell.replace(',',''))
            return cell

        def convert_year_to_int(cell, n):
            if isinstance(cell, str):
                # Remove the specific '(1)' characters
                cleaned_cell = cell.replace('(1)', '').strip()
                if len(cleaned_cell) == n:
                    return int(cleaned_cell)
            return cell
        
        n_value=7
        n_year=4

        df_sub = df_sub.map(lambda cell: convert_value_to_float(cell,n_value))
        df_sub = df_sub.map(lambda cell: convert_year_to_int(cell,n_year))

        df_sub = df_sub.apply(pd.to_numeric, errors='coerce')
        df_sub = df_sub.apply(lambda x: pd.Series(x.dropna().values), axis=1)

        df_sub.columns = df_sub.iloc[0].astype(int)
        df_sub = df_sub.iloc[1:].reset_index(drop=True)
        df_sub['Item'] = ['Mountain Reported EBITDA']

        melted = df_sub.melt(id_vars=['Item'],
                        var_name='Year', value_name='Amount')
            
        #melted = melted.iloc[1:]
        return melted

def check_for_duplicates_errors_revexp(df):
        '''This function should be run after combining multiple years of data.
    Because each financial form has 3 years, there are many duplicates in
    the combined dataframe. This function checks for duplicates and checks
    that all duplicates have identical values'''

        key_columns = ['Item', 'year'] # these are like header cols
        grouped = df.groupby(key_columns).nunique()
        #print('grouped: ', grouped)
        inconsistent_groups = grouped[(grouped > 1).any(axis=1)]
        if len(inconsistent_groups) == 0:
            print('No mismatches, looks all good')
            return None
            #assert len(inconsistent_groups) == 0, 'There are issues in the df. not all duplicates are identical'
        else:
            print('Mismatch found')
            target_val = 2 #indicates 2 unique vals in cell w/ same name, year
            counts = df.groupby(['Item','year']).nunique()

            stacked = counts.stack()

            bool_series = stacked == target_val

            aligned_bool_series = bool_series[bool_series]

            locations = aligned_bool_series.index.tolist()
            #locations = counts.stack()[df.stack() == target_val].index.tolist()
            items = [locations[i][0] for i in range(len(locations))]
            years = [locations[i][1] for i in range(len(locations))]
            column_name = [locations[i][2] for i in range(len(locations))]

            #make df 
            mismatch_df = pd.DataFrame({'Name':items,
                            'Year':years,
                            'Column':column_name})
            return mismatch_df
                
class CleanedRevenueExpTables:
    def __init__(self, raw_tables_dict, table_identifier):
        self.raw_tables_dict = raw_tables_dict.tables_dict
        self.data_dict = self.organize_table_data_runner()
        self.table_identifier = table_identifier
        self.mountain_ops_revenue, self.mountain_ops_expenses = self.process_data()

    def organize_table_data_runner(self):
        raw_tables_dict = self.raw_tables_dict
        df_ls, key_ls = [],[]
        for key in raw_tables_dict.keys():
            df = self.organize_table_data(raw_tables_dict[key])
            df_ls.append(df)
            key_ls.append(key)
        data_dict = dict(zip(key_ls, df_ls))
        return data_dict
    
    def clean_number(self, x):
        if pd.isna(x) or x == 'None':
            return np.nan
        if isinstance(x, str):
            # Skip text entries
            if any(c.isalpha() for c in x):
                return np.nan
            try:
                # Remove $, commas and spaces
                return float(x.replace('$', '').replace(',', '').strip())
            except ValueError:
                return np.nan
        return x
    
    def clean_mountain_revenue_df(self, df, year1, year2, year3):
        df = df.copy()
        numeric_cols = ['Column2', 'Column3', 'Column5', 'Column7', 'Column8', 'Column11']
        for col in numeric_cols:
            df[col] = df[col].apply(self.clean_number)
        result_df = pd.DataFrame({
            'Item': df['Column1'],
            f'{year1}': df['Column2'].combine_first(df['Column3']),
            f'{year2}': df['Column5'].combine_first(df['Column7']),
            f'{year3}': df['Column8'].combine_first(df['Column11'])
        })
        return result_df

    def organize_table_data(self, df_table):
        year1 = df_table.iloc[3, 1]
        #print('len year 1 ', len(year1))
        alt_year1 = df_table.iloc[2,1]
        #print('len alt year 1 ', len(alt_year1), ' , ', alt_year1)
        year2 = df_table.iloc[3, 3]
        alt_year2 = df_table.iloc[2,3]

        year3 = df_table.iloc[3, 5]
        alt_year3 = df_table.iloc[2,5]

        if len(year1) != 4:
            year1 = alt_year1
        #if len(year2) != 4:
            year2 = alt_year2
        #if len(year3) != 4:
            year3 = alt_year3
        #print(df_table.iloc[2])
        #print('year1 ', year1)
        #print('year2 ', year2)
        #print('year3 ', year3)
        df_table.columns = [f"Column{i}" for i in range(1, len(df_table.columns) + 1)]
        revenue_string = 'Total Mountain net revenue'
        last_row_of_revenue = df_table[df_table.iloc[:, 0] == revenue_string].index[0]
        total_expenses_string = 'Total Mountain operating expense'
        last_row_of_expenses = df_table[df_table.iloc[:, 0] == total_expenses_string].index[0]
        #header_rows = df_table.iloc[0:3]
        table_revenue = df_table.iloc[3:last_row_of_revenue + 1]
        table_costs = df_table.iloc[last_row_of_revenue + 1:last_row_of_expenses + 1]
        cleaned_revenue_df = self.clean_mountain_revenue_df(table_revenue, year1, year2, year3)
        cleaned_expenses_df = self.clean_mountain_revenue_df(table_costs, year1, year2, year3)
        return cleaned_revenue_df, cleaned_expenses_df
    
    def transform_df(self, df):
        return pd.melt(df, id_vars=['Item'], var_name='year', value_name='amount')

    def process_data(self):
        revenue_ls, expenses_ls = [], []
        years_ls = list(self.raw_tables_dict.keys())
        for key in self.data_dict.keys():
            #print(key)
            revenue, expenses = self.organize_table_data(self.raw_tables_dict[key])
            revenue_ls.append(revenue)
            expenses_ls.append(expenses)
        mountain_ops_revenue = dict(zip(years_ls, revenue_ls))
        mountain_ops_expenses = dict(zip(years_ls, expenses_ls))
        for key in mountain_ops_revenue:
            mountain_ops_revenue[key] = mountain_ops_revenue[key].dropna().reset_index(drop=True)
            mountain_ops_revenue[key] = self.transform_df(mountain_ops_revenue[key])
        for key in mountain_ops_expenses:
            mountain_ops_expenses[key] = mountain_ops_expenses[key].dropna().reset_index(drop=True)
            mountain_ops_expenses[key] = self.transform_df(mountain_ops_expenses[key])
        return mountain_ops_revenue, mountain_ops_expenses
        #self.mountain_ops_revenue
        #self.mountain_ops_expenses

    def combine_dfs(self):
        rev_df_full = pd.concat(self.mountain_ops_revenue.values(), ignore_index=True)
        exp_df_full = pd.concat(self.mountain_ops_expenses.values(), ignore_index=True)
        #switch units (units in table are in thousands)
        rev_df_full['amount'] = rev_df_full['amount'] * 1000
        exp_df_full['amount'] = exp_df_full['amount'] * 1000
        #format dates (a few have (1))
        rev_df_full['year'] = rev_df_full['year'].astype(str).str.extract(r'(\d{4})')[0]
        exp_df_full['year'] = exp_df_full['year'].astype(str).str.extract(r'(\d{4})')[0]
        #to datetime
        rev_df_full['year'] = pd.to_datetime(rev_df_full['year'], format='%Y').dt.year
        exp_df_full['year'] = pd.to_datetime(exp_df_full['year'], format='%Y').dt.year

        #now check for mismatcehs nd remove duplicates
        mismatches_rev = check_for_duplicates_errors_revexp(rev_df_full)
        mismatches_exp = check_for_duplicates_errors_revexp(exp_df_full)

        rev_df_full = rev_df_full.drop_duplicates(keep='first')
        exp_df_full = exp_df_full.drop_duplicates(keep='first')

        self.mismatches_rev = mismatches_rev
        self.mismatches_exp = mismatches_exp

        self.rev_df_full = rev_df_full
        self.exp_df_full = exp_df_full

def check_for_duplicates_errors_buyback(df):
        '''This function should be run after combining multiple years of data.
    Because each financial form has 3 years, there are many duplicates in
    the combined dataframe. This function checks for duplicates and checks
    that all duplicates have identical values'''

        key_columns = ['Year'] # these are like header cols
        grouped = df.groupby(key_columns).nunique()
        #print('grouped: ', grouped)
        inconsistent_groups = grouped[(grouped > 1).any(axis=1)]
        if len(inconsistent_groups) == 0:
            print('No mismatches, looks all good')
            return None
            #assert len(inconsistent_groups) == 0, 'There are issues in the df. not all duplicates are identical'
        else:
            print('Mismatch found')
            target_val = 2 #indicates 2 unique vals in cell w/ same name, year
            counts = df.groupby(['Year']).nunique()

            stacked = counts.stack()

            bool_series = stacked == target_val

            aligned_bool_series = bool_series[bool_series]

            locations = aligned_bool_series.index.tolist()
            #locations = counts.stack()[df.stack() == target_val].index.tolist()
            #items = [locations[i][0] for i in range(len(locations))]
            years = [locations[i][1] for i in range(len(locations))]
            column_name = [locations[i][2] for i in range(len(locations))]

            #make df 
            mismatch_df = pd.DataFrame({
                            'Year':years,
                            'Column':column_name})
            return mismatch_df
class CleanedBuybackTables:
    def __init__(self, raw_tables_dict, table_identifier):
        self.raw_buyback_dict = raw_tables_dict.tables_dict
        self.buyback_df, self.mismatches = self.clean_buyback_tables_driver()

    def clean_buyback_tables_driver(self):
        tables_dict = self.raw_buyback_dict
        key_ls, table_ls = [], []

        for key in tables_dict.keys():
            logging.info(f"Processing year: {key}")
            try:
                raw_df = tables_dict[key]
                clean_df = self.clean_buyback_tables(raw_df)
                #print(clean_df.columns)
                if clean_df is not None:
                    table_ls.append(clean_df)
                    key_ls.append(key)
                else:
                    logging.warning(f"No buyback data for year: {key}")
            except Exception as e:
                logging.error(f"Error processing year {key}: {e}", exc_info=True)

        # If no valid dataframes exist, return an empty DataFrame
        if not table_ls:
            logging.warning("No valid buyback data found in any year.")
            return pd.DataFrame()

        tables_dict = dict(zip(key_ls, table_ls))
        tables_df = pd.concat(tables_dict.values(), axis=0, ignore_index=True)
        mismatches = check_for_duplicates_errors_buyback(tables_df)
        tables_df.drop_duplicates(ignore_index=True, inplace=True)
        return tables_df, mismatches

    def clean_buyback_tables(self, df):
        try:
            df = df.copy()
            df.columns = [f"Column{i}" for i in range(1, len(df.columns) + 1)]
            df = df.replace('—', 0)

            match_index = df[df.apply(
                lambda row: row.astype(str).str.contains('Year Ended July 31,', case=False, na=False).any(), axis=1
            )].index

            # If a match is found, subset the DataFrame to rows after the match
            if not match_index.empty:
                df = df.loc[match_index[0] + 1:]  # Select rows after the first match
            else:
                logging.warning("No match found for 'Year Ended July 31,' in table.")
                return None

            df.iloc[0, 0] = 'Year'
            df.set_index('Column1', inplace=True)

            df_year = df.loc[['Year']]
            df_repurchase = df.loc[['Repurchases of common stock']]
            df_repurchase = df_repurchase.reset_index()

            value_columns = [f'Column{i}' for i in range(2, 7, 2)]
            buyback_data = df_repurchase.loc[
                df_repurchase['Column1'] == "Repurchases of common stock", value_columns
            ]

            if buyback_data.empty:
                logging.warning("Buyback data is empty after filtering.")
                return None

            buybacks = buyback_data.replace({r'\(|\)': '', ',': ''}, regex=True).astype(int).iloc[0]
            buyback_df = pd.DataFrame(buybacks).T

            new_df = pd.DataFrame({
                df_year['Column2'].values[0]: buyback_df['Column2'].values[0],
                df_year['Column3'].values[0]: buyback_df['Column4'].values[0],
                df_year['Column4'].values[0]: buyback_df['Column6'].values[0],
            }, index=[0])

            final_df = pd.melt(new_df, var_name='Year', value_name='Value')
            final_df['Value'] = final_df['Value'] * 1000
            return final_df

        except Exception as e:
            logging.error(f"Error cleaning buyback table: {e}", exc_info=True)
            return None

        