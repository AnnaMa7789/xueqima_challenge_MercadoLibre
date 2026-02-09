######################### GENAI ANALYSIS #########################
import sys
sys.path.append('../src')
import genai_analyzer as ga
import dashboard as dd
import os
import pandas as pd


pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None) 

scored_df = pd.read_csv('scored_df.csv')

## Title Analysis
results, enhanced_df = ga.run_title_improvement_pipeline(scored_df)

## Recommendation Generation
ai_results, enhanced_df=ga.run_non_title_issue_analysis(scored_df)

## Insight Synthesis
print("\n" + "="*80)
all_insights = ga.generate_multiple_insights(
        scored_df,
        analysis_types=["quality_patterns", "sales_drivers", "improvement_opportunities"]
    )

######################### DASHBOARD #########################
dd.run_dashboard(scored_df)
