import pandas as pd
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


## title quality

def analyze_title_quality(title):
    if pd.isna(title) or not isinstance(title, str): # check title value is valid
        return 0.0, {
            'info_density': 0.0,
            'key_elements': 0.0,
            'is_test_title': 0.0,
            'repetition_score': 0.0,
            'total_score': 0.0
        }
    
    title_str = str(title)
    quality_scores = {}

    # 1. info density
    words = title_str.lower().split()
    stop_words = {'para', 'de', 'y', 'con', 'por', 'para', 'el', 'la', 'los', 'las', 'del'}
    content_words = [w for w in words if w not in stop_words]
    quality_scores['info_density'] = len(content_words) / max(len(words), 1)
    
    # 2. key elements
    key_elements = 0
    if any(num in title_str for num in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']):
        key_elements += 1  #check if title has number info 
    if any(word in title_str.lower() for word in ['suspensor', 'short', 'camisa', 'bermuda']):
        key_elements += 1   #check if title has product type
    if any(word in title_str.lower() for word in ['tela', 'material', 'color', 'tamaño']):
        key_elements += 1   #check if title has description details
    
    quality_scores['key_elements'] = key_elements / 3.0
    
    # 3. test keyword
    test_indicators = ['testeo', 'prueba', 'no ofertar', 'item de', 'placeholder']
    quality_scores['is_test_title'] = 0 if any(ind in title_str.lower() for ind in test_indicators) else 1
    
    # 4. repeated words
    word_counts = {}
    for word in content_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    max_repeat = max(word_counts.values()) if word_counts else 1
    quality_scores['repetition_score'] = 1.0 if max_repeat <= 2 else 0.5
    
    # 综合分数
    weights = {
        'info_density': 0.3,
        'key_elements': 0.3,
        'is_test_title': 0.2,
        'repetition_score': 0.2
    }
    
    total_score = sum(quality_scores[k] * weights[k] for k in weights)
    quality_scores['total_score'] = total_score
    
    return total_score, quality_scores

def analyze_title_quality_distribution_and_correlation(df):
    """
    Analyze title quality distribution and correlation with sales rank
    """
    print("📊 TITLE QUALITY ANALYSIS")
    print("="*70)
    
    # 1. Calculate title quality scores
    print("1. Calculating title quality scores...")
    
    title_scores = []
    info_densities = []
    key_elements_scores = []
    is_test_titles = []
    repetition_scores = []
    
    for i, title in enumerate(df['title']):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} titles...")
        
        total_score, scores_dict = analyze_title_quality(title)
        
        title_scores.append(total_score)
        info_densities.append(scores_dict['info_density'])
        key_elements_scores.append(scores_dict['key_elements'])
        is_test_titles.append(scores_dict['is_test_title'])
        repetition_scores.append(scores_dict['repetition_score'])
    
    # Add scores to DataFrame
    df['title_score'] = title_scores
    df['title_info_density'] = info_densities
    df['title_key_elements'] = key_elements_scores
    df['title_is_test'] = is_test_titles
    df['title_repetition_score'] = repetition_scores
    
    # 2. Title quality distribution
    print("\n2. TITLE QUALITY DISTRIBUTION")
    print("-"*50)
    
    # Categorize title scores
    def categorize_title_score(score):
        if score >= 0.8:
            return 'Excellent (≥0.8)'
        elif score >= 0.6:
            return 'Good (0.6-0.79)'
        elif score >= 0.4:
            return 'Fair (0.4-0.59)'
        elif score >= 0.2:
            return 'Poor (0.2-0.39)'
        else:
            return 'Very Poor (<0.2)'
    
    df['title_quality_category'] = df['title_score'].apply(categorize_title_score)
    
    # Distribution of quality categories
    quality_dist = df['title_quality_category'].value_counts().sort_index()
    total_titles = len(df)
    
    print("\nTitle Quality Category Distribution:")
    for category, count in quality_dist.items():
        percentage = (count / total_titles) * 100
        print(f"  {category}: {count:,} titles ({percentage:.1f}%)")
    
    # Distribution of individual metrics
    print("\nIndividual Metric Statistics:")
    metrics = ['title_score', 'title_info_density', 'title_key_elements', 'title_is_test', 'title_repetition_score']
    for metric in metrics:
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        print(f"  {metric}: mean={mean_val:.3f}, median={median_val:.3f}, std={std_val:.3f}")
    
    # Test title detection
    test_title_count = (df['title_is_test'] == 0).sum()
    print(f"\nTest/Invalid Titles Detected: {test_title_count:,} ({(test_title_count/total_titles*100):.1f}%)")
    
    # 3. Calculate sales rank within category (only for categories with >10 products)
    print("\n3. CALCULATING SALES RANK WITHIN CATEGORIES...")
    
    # Filter categories with more than 10 products
    category_counts = df['category_id'].value_counts()
    valid_categories = category_counts[category_counts > 10].index.tolist()
    
    filtered_df = df[df['category_id'].isin(valid_categories)].copy()
    print(f"  Valid categories: {len(valid_categories)}")
    print(f"  Products in valid categories: {len(filtered_df):,}")
    
    # Calculate sales rank
    filtered_df['sales_rank'] = filtered_df.groupby('category_id')['sold_quantity'].rank(
        method='first', ascending=False
    )
    filtered_df['category_size'] = filtered_df.groupby('category_id')['id'].transform('count')
    filtered_df['percentile_rank'] = (filtered_df['sales_rank'] / filtered_df['category_size']) * 100
    
    # 4. Correlation analysis
    print("\n4. CORRELATION ANALYSIS: TITLE QUALITY VS SALES PERFORMANCE")
    print("-"*60)
    
    # Calculate correlations
    correlation_metrics = [
        'title_score',
        'title_info_density',
        'title_key_elements',
        'title_is_test',
        'title_repetition_score'
    ]
    
    correlation_results = []
    
    for metric in correlation_metrics:
        # Correlation with sales rank (negative correlation means better quality = better sales rank)
        corr_with_rank = filtered_df[metric].corr(filtered_df['sales_rank'])
        corr_with_percentile = filtered_df[metric].corr(filtered_df['percentile_rank'])
        corr_with_sales = filtered_df[metric].corr(filtered_df['sold_quantity'])
        
        correlation_results.append({
            'metric': metric,
            'corr_with_sales_rank': corr_with_rank,
            'corr_with_percentile_rank': corr_with_percentile,
            'corr_with_sold_quantity': corr_with_sales
        })
    
    # Display correlation results
    print("\nCorrelation Coefficients (Pearson):")
    print("(Note: Negative correlation with sales_rank/percentile_rank means better title quality = better sales)")
    print("\n{:<25} {:<20} {:<20} {:<20}".format(
        "Metric", "Corr with Sales Rank", "Corr with % Rank", "Corr with Sold Qty"
    ))
    print("-" * 85)
    
    for result in correlation_results:
        print("{:<25} {:<20.3f} {:<20.3f} {:<20.3f}".format(
            result['metric'],
            result['corr_with_sales_rank'],
            result['corr_with_percentile_rank'],
            result['corr_with_sold_quantity']
        ))
    
    # 5. Performance by title quality category
    print("\n5. SALES PERFORMANCE BY TITLE QUALITY CATEGORY")
    print("-"*60)
    
    performance_by_category = filtered_df.groupby('title_quality_category').agg({
        'id': 'count',
        'sales_rank': 'mean',
        'percentile_rank': 'mean',
        'sold_quantity': 'mean',
        'title_score': 'mean'
    }).round(3).reset_index()
    
    performance_by_category = performance_by_category.rename(columns={
        'id': 'product_count',
        'sales_rank': 'avg_sales_rank',
        'percentile_rank': 'avg_percentile_rank',
        'sold_quantity': 'avg_sold_quantity',
        'title_score': 'avg_title_score'
    })
    
    performance_by_category['percentage'] = (performance_by_category['product_count'] / len(filtered_df) * 100).round(1)
    performance_by_category = performance_by_category.sort_values('avg_percentile_rank')
    
    # Display in table format
    print("\n{:<20} {:<10} {:<12} {:<12} {:<12} {:<10}".format(
        "Quality Category", "Products", "Avg Rank", "Avg % Rank", "Avg Sold", "Avg Score"
    ))
    print("-" * 78)
    
    for idx, row in performance_by_category.iterrows():
        print("{:<20} {:<10,} {:<12.1f} {:<12.1f} {:<12.1f} {:<10.3f}".format(
            row['title_quality_category'],
            row['product_count'],
            row['avg_sales_rank'],
            row['avg_percentile_rank'],
            row['avg_sold_quantity'],
            row['avg_title_score']
        ))
    
    # 6. Statistical significance test
    print("\n6. STATISTICAL SIGNIFICANCE TEST")
    print("-"*60)
    
    # Compare top vs bottom quality groups
    top_group = filtered_df[filtered_df['title_quality_category'] == 'Excellent (≥0.8)']
    bottom_group = filtered_df[filtered_df['title_quality_category'] == 'Very Poor (<0.2)']
    
    if len(top_group) > 0 and len(bottom_group) > 0:
        top_avg_rank = top_group['percentile_rank'].mean()
        bottom_avg_rank = bottom_group['percentile_rank'].mean()
        rank_difference = bottom_avg_rank - top_avg_rank
        
        print(f"  Excellent titles (≥0.8): {len(top_group):,} products")
        print(f"    Avg percentile rank: {top_avg_rank:.1f}%")
        print(f"  Very poor titles (<0.2): {len(bottom_group):,} products")
        print(f"    Avg percentile rank: {bottom_avg_rank:.1f}%")
        print(f"  Difference: {rank_difference:.1f}% points")
        print(f"  Interpretation: Excellent titles rank {abs(rank_difference):.1f}% better than very poor titles")
    
    
    return df, filtered_df, correlation_results, performance_by_category
## get distribution of pictures
def safe_get_picture_info(df, pictures_col='pictures'):
    
    counts = []
    first_sizes = []
    first_max_sizes = []
    
    for idx, pics in enumerate(df[pictures_col]):
        if idx % 10000 == 0 and idx > 0:
            print(f"working on : {idx}/{len(df)}")
        
        count = 0
        first_size = None
        first_max_size = None
        
        try:
            # check null value 
            if pics is None or (isinstance(pics, float) and np.isnan(pics)):
                counts.append(0)
                first_sizes.append(None)
                first_max_sizes.append(None)
                continue
            
            # pic is list
            if isinstance(pics, list):
                count = len(pics)
                if count > 0:
                    first_pic = pics[0]
                    if isinstance(first_pic, dict):
                        first_size = first_pic.get('size')
                        first_max_size = first_pic.get('max_size')
            
            # pic is string
            elif isinstance(pics, str):
                pics_str = pics.strip()
                if not pics_str or pics_str == '[]' or pics_str == '{}':
                    count = 0
                else:
                    try:
                        parsed = json.loads(pics_str)
                        if isinstance(parsed, list):
                            count = len(parsed)
                            if count > 0 and isinstance(parsed[0], dict):
                                first_size = parsed[0].get('size')
                                first_max_size = parsed[0].get('max_size')
                        elif isinstance(parsed, dict):
                            count = 1
                            first_size = parsed.get('size')
                            first_max_size = parsed.get('max_size')
                    except json.JSONDecodeError:
                        if 'size' in pics_str and ('http' in pics_str or 'url' in pics_str):
                            count = 1
            
            # pic is dict
            elif isinstance(pics, dict):
                count = 1
                first_size = pics.get('size')
                first_max_size = pics.get('max_size')
            
        except Exception as e:
            print(f"error")
            count = 0
            first_size = None
            first_max_size = None
        
        counts.append(count)
        first_sizes.append(first_size)
        first_max_sizes.append(first_max_size)
    
    return counts, first_sizes, first_max_sizes
def safe_analyze_attributes(df, attributes_col='attributes'):
    print("🔍 Starting safe attribute analysis...")
    
    n = len(df)
    entry_counts = []
    empty_counts = []
    total_counts = []
    
    for idx in range(n):
        # Show progress for large datasets
        if idx > 0 and idx % 10000 == 0:
            print(f"  Processed {idx:,} rows...")
        
        # Get the attribute value for this row
        attr_value = df[attributes_col].iloc[idx]
        
        # Initialize counters for this row
        entry_count = 0
        empty_count = 0
        total_count = 0
        
        try:
            # Step 1: Check for None/NaN
            if attr_value is None:
                entry_counts.append(0)
                empty_counts.append(0)
                total_counts.append(0)
                continue
            
            # Step 2: Check for pandas NaN (float type)
            if isinstance(attr_value, float) and pd.isna(attr_value):
                entry_counts.append(0)
                empty_counts.append(0)
                total_counts.append(0)
                continue
            
            # Step 3: Check for empty string
            if isinstance(attr_value, str) and not attr_value.strip():
                entry_counts.append(0)
                empty_counts.append(0)
                total_counts.append(0)
                continue
            
            # Step 4: Parse the value
            parsed_data = None
            
            # If it's already a list
            if isinstance(attr_value, list):
                parsed_data = attr_value
            
            # If it's a string, try to parse as JSON
            elif isinstance(attr_value, str):
                attr_str = attr_value.strip()
                
                # Skip empty JSON structures
                if attr_str == '[]' or attr_str == '{}':
                    entry_counts.append(0)
                    empty_counts.append(0)
                    total_counts.append(0)
                    continue
                
                try:
                    parsed = json.loads(attr_str)
                    if isinstance(parsed, list):
                        parsed_data = parsed
                    elif isinstance(parsed, dict):
                        parsed_data = [parsed]  # Wrap single dict in list
                    else:
                        parsed_data = []
                except json.JSONDecodeError:
                    # Not valid JSON, treat as no attributes
                    parsed_data = []
            
            # If it's a dict (single attribute)
            elif isinstance(attr_value, dict):
                parsed_data = [attr_value]
            
            # Unknown type
            else:
                parsed_data = []
            
            # Step 5: Analyze the parsed data
            if parsed_data is not None and isinstance(parsed_data, list):
                entry_count = len(parsed_data)
                
                # Count fields in each attribute entry
                for attr_entry in parsed_data:
                    if isinstance(attr_entry, dict):
                        for key, value in attr_entry.items():
                            total_count += 1
                            
                            # Check if field is empty
                            if (value is None or 
                                value == '' or 
                                (isinstance(value, str) and not value.strip()) or
                                (isinstance(value, float) and pd.isna(value))):
                                empty_count += 1
        
        except Exception as e:
            # If any error occurs, use default values
            # Uncomment for debugging: print(f"Error at index {idx}: {e}")
            entry_count = 0
            empty_count = 0
            total_count = 0
        
        # Append results for this row
        entry_counts.append(entry_count)
        empty_counts.append(empty_count)
        total_counts.append(total_count)
    
    print(f"✅ Analysis complete! Processed {n:,} rows")
    return entry_counts, empty_counts, total_counts
    
# Calculate completeness percentage (avoiding division by zero)
def calculate_completeness(row):
    if row['attr_total_fields'] > 0:
        return round((row['attr_total_fields'] - row['attr_empty_fields']) / row['attr_total_fields'] * 100, 2)
    return 0.0

# Calculate missing fields format: "X fields missing in Y fields"
def format_missing_fields(row):
    """Format as 'X fields missing in Y fields'"""
    total_fields = row['attr_total_fields']
    empty_fields = row['attr_empty_fields']
    filled_fields = total_fields - empty_fields
    
    if total_fields == 0:
        return "0 fields (no data)"
    
    return f"{empty_fields} missing in {total_fields} fields ({filled_fields} filled)"


def analyze_attribute_impact_on_sales(df):
    """
    Analyze impact of attribute entry count and completeness on sales ranking
    Only consider categories with more than 10 products
    """
    
    print("📊 ANALYZING ATTRIBUTE IMPACT ON SALES RANKING")
    print("="*70)
    
    # 1. Filter categories with more than 10 products
    print("Filtering categories with more than 10 products...")
    category_counts = df['category_id'].value_counts()
    valid_categories = category_counts[category_counts > 10].index.tolist()
    
    filtered_df = df[df['category_id'].isin(valid_categories)].copy()
    print(f"  Valid categories: {len(valid_categories)}")
    print(f"  Products in valid categories: {len(filtered_df):,}")
    
    # 2. Calculate sales rank within each category
    print("\nCalculating sales rankings within categories...")
    filtered_df['sales_rank'] = filtered_df.groupby('category_id')['sold_quantity'].rank(
        method='first', ascending=False
    )
    filtered_df['category_size'] = filtered_df.groupby('category_id')['id'].transform('count')
    filtered_df['percentile_rank'] = (filtered_df['sales_rank'] / filtered_df['category_size']) * 100
    
    # 3. Group entry counts (combine 5+ as one group)
    print("\nGrouping attribute entry counts...")
    filtered_df['entry_group'] = filtered_df['attr_entries'].apply(
        lambda x: '5+' if x >= 5 else str(x)
    )
    
    # 4. Group completeness percentages
    print("Grouping completeness percentages...")
    def categorize_completeness(pct):
        if pct == 100:
            return '100% (Perfect)'
        elif pct >= 90:
            return '90-99%'
        elif pct >= 80:
            return '80-89%'
        elif pct >= 70:
            return '70-79%'
        elif pct >= 50:
            return '50-69%'
        elif pct > 0:
            return '1-49%'
        else:
            return '0%'
    
    filtered_df['completeness_group'] = filtered_df['attr_completeness_pct'].apply(categorize_completeness)
    
    # 5. Analyze each group
    print("\nAnalyzing impact by entry count and completeness...")
    
    # Group by both factors
    grouped = filtered_df.groupby(['entry_group', 'completeness_group'])
    
    analysis_results = []
    
    for (entry_group, completeness_group), group_df in grouped:
        if len(group_df) < 10:  # Skip groups with too few products
            continue
            
        result = {
            'entry_group': entry_group,
            'completeness_group': completeness_group,
            'product_count': len(group_df),
            'avg_sales_rank': group_df['sales_rank'].mean(),
            'avg_percentile_rank': group_df['percentile_rank'].mean(),
            'avg_sold_quantity': group_df['sold_quantity'].mean(),
            'pct_top_10': (group_df['sales_rank'] <= 10).mean() * 100,
            'pct_top_25_percent': (group_df['percentile_rank'] <= 25).mean() * 100
        }
        analysis_results.append(result)
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(analysis_results)
    results_df['percentage_of_total'] = (results_df['product_count'] / len(filtered_df) * 100).round(2)
    
    # Round numeric columns
    numeric_cols = ['avg_sales_rank', 'avg_percentile_rank', 'avg_sold_quantity', 'pct_top_10', 'pct_top_25_percent']
    results_df[numeric_cols] = results_df[numeric_cols].round(2)
    
    # Sort by avg_percentile_rank (lower = better sales)
    results_df = results_df.sort_values('avg_percentile_rank')
    
    return filtered_df, results_df

def analyze_shipping_field(shipping_value):
    """
    Analyze a single shipping field value
    
    Returns:
        dict: Extracted shipping information
    """
    result = {
        'has_shipping_data': False,
        'local_pick_up': None,
        'free_shipping': None,
        'mode': None,
        'dimensions': None,
        'methods_count': 0,
        'tags_count': 0
    }
    
    # Check if shipping_value is None or empty
    if shipping_value is None:
        return result
    
    # Handle different data types
    shipping_dict = None
    
    if isinstance(shipping_value, dict):
        shipping_dict = shipping_value
    elif isinstance(shipping_value, str):
        try:
            shipping_dict = json.loads(shipping_value)
        except (json.JSONDecodeError, TypeError):
            # Try to parse as string representation
            try:
                # Remove single quotes if present
                cleaned = shipping_value.replace("'", '"')
                shipping_dict = json.loads(cleaned)
            except:
                return result
    else:
        return result
    
    # Ensure it's a dictionary
    if not isinstance(shipping_dict, dict):
        return result
    
    # Extract fields
    result['has_shipping_data'] = True
    
    # Get local_pick_up
    result['local_pick_up'] = shipping_dict.get('local_pick_up')
    
    # Get free_shipping
    result['free_shipping'] = shipping_dict.get('free_shipping')
    
    # Get mode
    result['mode'] = shipping_dict.get('mode')
    
    # Get dimensions
    result['dimensions'] = shipping_dict.get('dimensions')
    
    # Get methods count
    methods = shipping_dict.get('methods')
    if isinstance(methods, list):
        result['methods_count'] = len(methods)
    elif methods is not None:
        result['methods_count'] = 1
    
    # Get tags count
    tags = shipping_dict.get('tags')
    if isinstance(tags, list):
        result['tags_count'] = len(tags)
    elif tags is not None:
        result['tags_count'] = 1
    
    return result
def comprehensive_factor_correlation_analysis(df):
    """
    Comprehensive correlation analysis of multiple factors with sales performance
    All factors coded as positive (higher value = better listing quality)
    """
    print("📊 COMPREHENSIVE FACTOR CORRELATION ANALYSIS")
    print("="*80)
    
    # 1. Prepare the data with all factors
    print("1. Preparing data with all factors...")
    
    # Create a working copy
    analysis_df = df.copy()
    
    # 2. Calculate sales rank within categories (only categories with >10 products)
    print("2. Calculating sales rankings...")
    
    # Filter categories with more than 10 products
    category_counts = analysis_df['category_id'].value_counts()
    valid_categories = category_counts[category_counts > 10].index.tolist()
    analysis_df = analysis_df[analysis_df['category_id'].isin(valid_categories)].copy()
    
    print(f"  Valid categories: {len(valid_categories)}")
    print(f"  Products analyzed: {len(analysis_df):,}")
    
    # Calculate sales rank
    analysis_df['sales_rank'] = analysis_df.groupby('category_id')['sold_quantity'].rank(
        method='first', ascending=False
    )
    analysis_df['category_size'] = analysis_df.groupby('category_id')['id'].transform('count')
    analysis_df['percentile_rank'] = (analysis_df['sales_rank'] / analysis_df['category_size']) * 100
    
    # 3. Calculate all factors - all as positive factors
    print("3. Calculating all factors (all as positive indicators)...")
    
    # Factor 1: Title length (in characters) - POSITIVE FACTOR
    analysis_df['title_length'] = analysis_df['title'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    
    # Factor 2: Title quality score - POSITIVE FACTOR
    if 'title_score' not in analysis_df.columns:
        print("  Calculating title quality scores...")
        title_scores = []
        for title in analysis_df['title']:
            score, _ = analyze_title_quality(title)
            title_scores.append(score)
        analysis_df['title_score'] = title_scores
    
    # Factor 3: HAS video (1 = has video, 0 = missing) - POSITIVE FACTOR
    analysis_df['has_video'] = analysis_df['video_id'].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    # Factor 4: HAS been updated (1 = updated, 0 = never updated) - POSITIVE FACTOR
    analysis_df['has_updated'] = analysis_df.apply(
        lambda row: 0 if (pd.notna(row['last_updated']) and pd.notna(row['date_created']) and 
                         str(row['last_updated'])[:10] == str(row['date_created'])[:10]) else 1,
        axis=1
    )
    
    # Factor 5: Number of pictures - POSITIVE FACTOR
    if 'picture_count' not in analysis_df.columns:
        print("  Warning: picture_count not found, using placeholder")
        analysis_df['picture_count'] = 0
    
    # Factor 6: Attributes entry number - POSITIVE FACTOR
    if 'attr_entries' not in analysis_df.columns:
        print("  Warning: attr_entries not found, using placeholder")
        analysis_df['attr_entries'] = 0
    
    # Factor 7: Attributes completeness percentage - POSITIVE FACTOR
    if 'attr_completeness_pct' not in analysis_df.columns:
        print("  Warning: attr_completeness_pct not found, using placeholder")
        analysis_df['attr_completeness_pct'] = 0
    
    # 4. Correlation analysis
    print("4. Calculating correlations...")
    
    # Define factors for correlation
    factors = {
        'title_length': 'Title Length (characters)',
        'title_score': 'Title Quality Score',
        'has_video': 'Has Video',
        'has_updated': 'Has Updated',
        'picture_count': 'Number of Pictures',
        'attr_entries': 'Attributes Entry Count',
        'attr_completeness_pct': 'Attributes Completeness %'
    }
    
    # Performance metrics
    performance_metrics = {
        'sales_rank': 'Sales Rank (lower = better)',
        'percentile_rank': 'Percentile Rank (lower = better)',
        'sold_quantity': 'Sold Quantity'
    }
    
    # Calculate correlation matrix
    all_columns = list(factors.keys()) + list(performance_metrics.keys())
    correlation_matrix = analysis_df[all_columns].corr()
    
    # 5. Display correlation results
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: FACTORS VS SALES PERFORMANCE")
    print("All factors coded as positive (higher = better listing quality)")
    print("Negative correlation = better factor → better sales performance")
    print("="*80)
    
    print("\n📈 Correlation Matrix:")
    print("-"*80)
    
    # Display correlations in a clean table
    print("\n{:<30} {:<15} {:<15} {:<15}".format(
        "Factor", "Corr with Sales Rank", "Corr with % Rank", "Corr with Sold Qty"
    ))
    print("-"*75)
    
    correlation_results = []
    
    for factor_key, factor_name in factors.items():
        if factor_key in correlation_matrix.index:
            corr_rank = correlation_matrix.loc[factor_key, 'sales_rank']
            corr_percentile = correlation_matrix.loc[factor_key, 'percentile_rank']
            corr_sales = correlation_matrix.loc[factor_key, 'sold_quantity']
            
            correlation_results.append({
                'factor': factor_name,
                'factor_key': factor_key,
                'corr_sales_rank': corr_rank,
                'corr_percentile_rank': corr_percentile,
                'corr_sold_quantity': corr_sales
            })
            
            print("{:<30} {:<15.3f} {:<15.3f} {:<15.3f}".format(
                factor_name[:30],
                corr_rank,
                corr_percentile,
                corr_sales
            ))
    
    # 6. Factor importance ranking
    print("\n" + "="*80)
    print("FACTOR IMPORTANCE RANKING")
    print("Sorted by impact on sales performance (absolute correlation)")
    print("="*80)
    
    # Sort by absolute correlation with sales rank
    correlation_results_sorted = sorted(
        correlation_results, 
        key=lambda x: abs(x['corr_sales_rank']), 
        reverse=True
    )
    
    print("\n{:<5} {:<30} {:<15} {:<25}".format(
        "Rank", "Factor", "Correlation", "Impact Level"
    ))
    print("-"*80)
    
    for i, result in enumerate(correlation_results_sorted, 1):
        corr = result['corr_sales_rank']
        
        # Determine impact level
        if corr < -0.15:
            impact = "🚀 STRONG POSITIVE"
        elif corr < -0.1:
            impact = "✅ STRONG POSITIVE"
        elif corr < -0.05:
            impact = "📈 MODERATE POSITIVE"
        elif corr < 0:
            impact = "↗️  WEAK POSITIVE"
        elif corr < 0.05:
            impact = "↘️  WEAK NEGATIVE"
        elif corr < 0.1:
            impact = "📉 MODERATE NEGATIVE"
        else:
            impact = "❌ STRONG NEGATIVE"
        
        print("{:<5} {:<30} {:<15.3f} {:<25}".format(
            i,
            result['factor'][:30],
            corr,
            impact
        ))
    
    return analysis_df, correlation_results

def comprehensive_price_correlation_analysis(df, price_col='price', 
                                           sold_col='sold_quantity',
                                           category_col='category_id',
                                           id_col='id'):
    """
    Comprehensive correlation analysis of price relative position with sales performance
    Analyzes price position within categories and its relationship with sales
    """
    print("💰 COMPREHENSIVE PRICE POSITION ANALYSIS")
    print("="*80)
    
    # 1. Prepare the data
    print("1. Preparing data...")
    
    # Create a working copy
    analysis_df = df.copy()
    
    # 2. Filter categories with >10 products
    print("2. Filtering categories...")
    category_counts = analysis_df[category_col].value_counts()
    valid_categories = category_counts[category_counts > 10].index.tolist()
    analysis_df = analysis_df[analysis_df[category_col].isin(valid_categories)].copy()
    
    print(f"  Valid categories: {len(valid_categories)}")
    print(f"  Products analyzed: {len(analysis_df):,}")
    
    # 3. Calculate price and sales relative positions
    print("3. Calculating relative positions within categories...")
    
    # Group by category
    grouped = analysis_df.groupby(category_col)
    
    # Calculate category sizes
    analysis_df['category_size'] = grouped[id_col].transform('count')
    
    # Calculate price percentile within category (0-100)
    # 0% = cheapest in category, 100% = most expensive in category
    analysis_df['price_percentile'] = grouped[price_col].rank(
        pct=True, method='average'
    ) * 100
    
    # Calculate sales percentile within category (0-100)
    # 0% = lowest sales in category, 100% = highest sales in category
    analysis_df['sales_percentile'] = grouped[sold_col].rank(
        pct=True, method='average'
    ) * 100
    
    # Calculate price rank (1 = cheapest, higher = more expensive)
    analysis_df['price_rank'] = grouped[price_col].rank(
        method='first', ascending=True
    )
    
    # Calculate sales rank (1 = highest sales, higher = lower sales)
    analysis_df['sales_rank'] = grouped[sold_col].rank(
        method='first', ascending=False
    )
    
    # Calculate performance score: sales_percentile - price_percentile
    # Positive = selling better than price position would suggest
    # Negative = selling worse than price position would suggest
    analysis_df['price_performance_score'] = (
        analysis_df['sales_percentile'] - analysis_df['price_percentile']
    )
    
    # 4. Create price position categories
    print("4. Creating price position categories...")
    
    def categorize_price_position(percentile):
        if percentile <= 25:
            return 'Bottom 25% (Low Price)'
        elif percentile <= 50:
            return '25-50% (Lower Mid Price)'
        elif percentile <= 75:
            return '50-75% (Upper Mid Price)'
        else:
            return 'Top 25% (High Price)'
    
    analysis_df['price_position'] = analysis_df['price_percentile'].apply(
        categorize_price_position
    )
    
    # 5. Calculate correlation metrics
    print("5. Calculating correlation metrics...")
    
    # Define metrics for correlation analysis
    price_metrics = {
        'price': 'Absolute Price',
        'price_percentile': 'Price Percentile in Category',
        'price_rank': 'Price Rank (1=cheapest)',
        'price_performance_score': 'Price Performance Score'
    }
    
    sales_metrics = {
        'sold_quantity': 'Sold Quantity',
        'sales_percentile': 'Sales Percentile in Category',
        'sales_rank': 'Sales Rank (1=best)'
    }
    
    # Calculate correlation matrix
    all_columns = list(price_metrics.keys()) + list(sales_metrics.keys())
    correlation_matrix = analysis_df[all_columns].corr()
    
    # 6. Display correlation results
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: PRICE METRICS VS SALES PERFORMANCE")
    print("Price Percentile: 0% = cheapest in category, 100% = most expensive in category")
    print("Sales Rank: 1 = highest sales in category")
    print("="*80)
    
    print("\n📈 Correlation Matrix:")
    print("-"*80)
    
    print("\n{:<35} {:<20} {:<20} {:<20}".format(
        "Price Metric", "Corr with Sales Rank", "Corr with Sales %", "Corr with Sold Qty"
    ))
    print("-"*95)
    
    correlation_results = []
    
    for metric_key, metric_name in price_metrics.items():
        if metric_key in correlation_matrix.index:
            corr_rank = correlation_matrix.loc[metric_key, 'sales_rank']
            corr_percentile = correlation_matrix.loc[metric_key, 'sales_percentile']
            corr_sales = correlation_matrix.loc[metric_key, 'sold_quantity']
            
            correlation_results.append({
                'metric': metric_name,
                'metric_key': metric_key,
                'corr_sales_rank': corr_rank,
                'corr_sales_percentile': corr_percentile,
                'corr_sold_quantity': corr_sales
            })
            
            print("{:<35} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                metric_name[:35],
                corr_rank,
                corr_percentile,
                corr_sales
            ))
    
    # 7. Price position performance analysis
    print("\n" + "="*80)
    print("PRICE POSITION PERFORMANCE ANALYSIS")
    print("Average sales performance by price position")
    print("="*80)
    
    position_performance = analysis_df.groupby('price_position').agg({
        'price_percentile': 'mean',
        'sales_percentile': 'mean',
        'price_performance_score': 'mean',
        id_col: 'count'
    }).rename(columns={id_col: 'product_count'}).round(2)
    
    position_performance = position_performance.sort_values('price_percentile')
    
    print("\n{:<30} {:<15} {:<15} {:<20} {:<15}".format(
        "Price Position", "Avg Price %", "Avg Sales %", "Performance Score", "Product Count"
    ))
    print("-"*95)
    
    for position, row in position_performance.iterrows():
        perf_score = row['price_performance_score']
        
        # Performance indicator
        if perf_score > 10:
            indicator = "🚀 EXCELLENT"
        elif perf_score > 5:
            indicator = "✅ GOOD"
        elif perf_score > 0:
            indicator = "↗️  POSITIVE"
        elif perf_score > -5:
            indicator = "↘️  NEUTRAL"
        elif perf_score > -10:
            indicator = "⚠️  BELOW EXPECTED"
        else:
            indicator = "❌ POOR"
        
        print("{:<30} {:<15.1f} {:<15.1f} {:<10.1f} {:<10} {:<15}".format(
            position[:30],
            row['price_percentile'],
            row['sales_percentile'],
            perf_score,
            indicator,
            int(row['product_count'])
        ))
    
    # 8. Price importance ranking
    print("\n" + "="*80)
    print("PRICE IMPORTANCE SUMMARY")
    print("Key insights on price impact on sales performance")
    print("="*80)
    
    # Calculate overall correlation
    overall_corr = analysis_df['price_percentile'].corr(analysis_df['sales_percentile'])
    
    print(f"\n📊 Overall Price-Sales Correlation: {overall_corr:.3f}")
    
    if overall_corr < -0.2:
        print("💡 INSIGHT: Strong negative correlation - Price position has significant impact on sales")
    elif overall_corr < -0.1:
        print("💡 INSIGHT: Moderate negative correlation - Price position matters")
    elif overall_corr < 0:
        print("💡 INSIGHT: Weak negative correlation - Price has some influence")
    elif overall_corr == 0:
        print("💡 INSIGHT: No correlation - Price position doesn't affect sales")
    else:
        print(f"💡 INSIGHT: Positive correlation ({overall_corr:.3f}) - Higher price positions correlate with better sales")
    
    # Performance by price position
    print(f"\n🏆 Best Performing Price Position: {position_performance['price_performance_score'].idxmax()}")
    print(f"📉 Worst Performing Price Position: {position_performance['price_performance_score'].idxmin()}")
    
    # Price sensitivity analysis
    avg_perf_score = analysis_df['price_performance_score'].mean()
    print(f"\n📈 Average Price Performance Score: {avg_perf_score:.2f}")
    
    if avg_perf_score > 5:
        print("✅ Products generally sell better than their price position suggests")
    elif avg_perf_score > 0:
        print("↗️  Products slightly outperform their price position")
    elif avg_perf_score > -5:
        print("↘️  Products perform as expected for their price position")
    else:
        print("⚠️  Products generally underperform their price position")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return analysis_df, correlation_results


# Example usage
def run_price_analysis(df):
    """
    Run the comprehensive price analysis
    """
    print("🔍 Starting Price Position Analysis...")
    print("="*80)
    
    # Run the analysis
    analysis_df, correlation_results = comprehensive_price_correlation_analysis(
        df,
        price_col='price',
        sold_col='sold_quantity',
        category_col='category_id',
        id_col='id'
    )
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"   Analyzed {len(analysis_df):,} products")
    print(f"   Across {analysis_df['category_id'].nunique():,} categories")
    
    return analysis_df, correlation_results



def calculate_dynamic_weights(price_correlation_results, factor_correlation_results):
    """
    Calculate dynamic weights for listing quality dimensions based on correlation analysis.
    Weights are proportional to absolute correlation with sales rank.
    
    Parameters:
    -----------
    price_correlation_results : list of dict
        Results from comprehensive_price_correlation_analysis
    factor_correlation_results : list of dict
        Results from comprehensive_factor_correlation_analysis
        
    Returns:
    --------
    dict : Weights for each dimension (sum to 100)
    """
    
    # Combine all factors - only include 'Price Percentile in Category' from price results
    all_factors = []
    
    # Add only 'Price Percentile in Category' factor from price analysis
    for result in price_correlation_results:
        if result['metric'] == 'Price Percentile in Category':
            all_factors.append({
                'name': 'Price Position in Category',
                'corr_sales_rank': abs(result['corr_sales_rank'])
            })
            break  # Only need this one factor
    
    # Add listing quality factors
    for result in factor_correlation_results:
        all_factors.append({
            'name': result['factor'],
            'corr_sales_rank': abs(result['corr_sales_rank'])
        })
    
    # Calculate total impact
    total_impact = sum(factor['corr_sales_rank'] for factor in all_factors)
    
    # Calculate weights (proportional to impact)
    weights = {}
    for factor in all_factors:
        raw_weight = (factor['corr_sales_rank'] / total_impact) * 100
        weights[factor['name']] = round(raw_weight, 2)
    
    return weights


def calculate_listing_quality_score(analysis_df, weights):
    """
    Calculate comprehensive listing quality score (0-100) based on analysis_df and dynamic weights.
    
    Parameters:
    -----------
    analysis_df : pandas DataFrame
        DataFrame containing all calculated metrics from previous analyses
    weights : dict
        Weights from calculate_dynamic_weights function
        
    Returns:
    --------
    pandas DataFrame
        Original DataFrame with added quality scores and breakdown
    """
    
    # Create a working copy
    result_df = analysis_df.copy()
    
    # Initialize score columns
    result_df['quality_score'] = 0.0
    result_df['score_breakdown'] = None
    
    print(f"Calculating listing quality scores for {len(result_df)} products...")
    print(f"Using weights: {weights}")
    
    # Helper functions for each dimension score calculation
    def calculate_title_completeness_score(title_length):
        """Calculate score based on title length"""
        if pd.isna(title_length):
            return 0
        
        # Score based on title length
        if title_length >= 80:
            return 90
        elif title_length >= 50:
            return 75
        elif title_length >= 30:
            return 60
        elif title_length >= 15:
            return 40
        else:
            return 20
    
    def calculate_title_quality_score(title_score):
        """Calculate score based on title quality score"""
        if pd.isna(title_score):
            return 0
        return min(float(title_score)*100, 100)
    
    def calculate_video_content_score(has_video):
        """Calculate score for video presence"""
        if pd.isna(has_video):
            return 0
        return 100 if has_video == 1 else 0
    
    def calculate_listing_freshness_score(has_updated):
        """Calculate score based on listing update status"""
        if pd.isna(has_updated):
            return 30
        return 80 if has_updated == 1 else 30
    
    def calculate_image_quality_score(picture_count):
        """Calculate score based on number of images"""
        if pd.isna(picture_count):
            return 0
        
        # Score based on image count
        if picture_count >= 8:
            return 100
        elif picture_count >= 5:
            return 80
        elif picture_count >= 3:
            return 70
        elif picture_count >= 1:
            return 50
        else:
            return 0
    
    def calculate_attributes_quantity_score(attr_entries):
        """Calculate score based on number of attributes"""
        if pd.isna(attr_entries):
            return 0
        
        # Score based on attributes count
        if attr_entries >= 15:
            return 100
        elif attr_entries >= 10:
            return 80
        elif attr_entries >= 5:
            return 60
        elif attr_entries >= 1:
            return 40
        else:
            return 0
    
    def calculate_attributes_completeness_score(attr_completeness_pct):
        """Calculate score based on attributes completeness percentage"""
        if pd.isna(attr_completeness_pct):
            return 0
        return min(float(attr_completeness_pct), 100)
    
    def calculate_price_position_score(price_percentile):
        """Calculate score based on price percentile in category"""
        if pd.isna(price_percentile):
            return 50
        
        # Invert percentile: lower price (lower percentile) gets higher score
        # 0% percentile (cheapest) = 100 score
        # 100% percentile (most expensive) = 0 score
        return 100 - price_percentile
    
    # Create dimension mapping for easier access
    dimension_mapping = {
        'Title Length (characters)': {
            'function': calculate_title_completeness_score,
            'data_column': 'title_length'  # Need to calculate this if not exists
        },
        'Title Quality Score': {
            'function': calculate_title_quality_score,
            'data_column': 'title_score'
        },
        'Has Video': {
            'function': calculate_video_content_score,
            'data_column': 'has_video'
        },
        'Has Updated': {
            'function': calculate_listing_freshness_score,
            'data_column': 'has_updated'
        },
        'Number of Pictures': {
            'function': calculate_image_quality_score,
            'data_column': 'picture_count'
        },
        'Attributes Entry Count': {
            'function': calculate_attributes_quantity_score,
            'data_column': 'attr_entries'
        },
        'Attributes Completeness %': {
            'function': calculate_attributes_completeness_score,
            'data_column': 'attr_completeness_pct'
        },
        'Price Position in Category': {
            'function': calculate_price_position_score,
            'data_column': 'price_percentile'
        }
    }
    
    # Calculate any missing required columns
    # 1. title_length
    if 'title_length' not in result_df.columns and 'title' in result_df.columns:
        print("Calculating title lengths...")
        result_df['title_length'] = result_df['title'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
    
    # 2. has_video
    if 'has_video' not in result_df.columns and 'video_id' in result_df.columns:
        print("Calculating video presence...")
        result_df['has_video'] = result_df['video_id'].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
        )
    
    # 3. has_updated
    if 'has_updated' not in result_df.columns and 'last_updated' in result_df.columns and 'date_created' in result_df.columns:
        print("Calculating update status...")
        result_df['has_updated'] = result_df.apply(
            lambda row: 0 if (pd.notna(row['last_updated']) and pd.notna(row['date_created']) and 
                             str(row['last_updated'])[:10] == str(row['date_created'])[:10]) else 1,
            axis=1
        )
    
    # Calculate scores for each product
    all_breakdowns = []
    
    for idx, row in result_df.iterrows():
        score_breakdown = {}
        total_score = 0
        
        # Calculate each dimension score based on available weights
        for weight_key, weight_value in weights.items():
            if weight_key not in dimension_mapping:
                print(f"Warning: Weight key '{weight_key}' not found in dimension mapping")
                continue
            
            mapping = dimension_mapping[weight_key]
            score_func = mapping['function']
            data_column = mapping['data_column']
            
            # Check if data column exists
            if data_column not in row:
                print(f"Warning: Data column '{data_column}' not found for {weight_key}")
                raw_score = 50  # Default score
            else:
                try:
                    data = row[data_column]
                    raw_score = score_func(data)
                except Exception as e:
                    print(f"Warning: Error calculating {weight_key} for product {row.get('id', idx)}: {e}")
                    raw_score = 50  # Default score
            
            # Calculate weighted score
            weighted_score = raw_score * (weight_value / 100)
            total_score += weighted_score
            
            # Store scores
            score_breakdown[weight_key] = {
                'raw_score': raw_score,
                'weight': weight_value,
                'weighted_score': weighted_score,
                'data_value': row.get(data_column, None) if data_column in row else None
            }
        
        # Store in results
        result_df.at[idx, 'quality_score'] = round(total_score, 2)
        result_df.at[idx, 'score_breakdown'] = score_breakdown
        all_breakdowns.append(score_breakdown)
        
        # Show progress for large datasets
        if len(result_df) > 10000 and (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(result_df)} products...")
    
    print(f"Completed calculating scores for {len(result_df)} products.")
    
    return result_df


def identify_high_priority_listings(scored_df, category_col='category_id', 
                                   quality_score_col='quality_score',
                                   price_col='price',
                                   sales_col='sold_quantity',
                                   id_col='id'):
    """
    Identify high-priority listings: low quality + high sales potential.
    
    Criteria:
    1. Quality score in bottom 25th percentile within its category
    2. Category has above-average price and sales quantity
    3. Individual product has decent sales (not bottom performers)
    
    Parameters:
    -----------
    scored_df : pandas DataFrame
        DataFrame with quality scores and other metrics
    category_col : str
        Column name for category IDs
    quality_score_col : str
        Column name for quality scores
    price_col : str
        Column name for price data
    sales_col : str
        Column name for sales quantity data
    id_col : str
        Column name for product IDs
        
    Returns:
    --------
    pandas DataFrame : Filtered DataFrame with high-priority listings
    dict : Analysis summary
    """
    
    print("🔍 IDENTIFYING HIGH-PRIORITY LISTINGS")
    print("="*80)
    
    # Create a working copy
    df = scored_df.copy()
    
    # Step 1: Calculate category-level metrics
    print("1. Calculating category-level metrics...")
    
    category_stats = df.groupby(category_col).agg({
        price_col: ['mean', 'median', 'count'],
        sales_col: ['mean', 'median', 'sum'],
        quality_score_col: ['mean', 'median', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    category_stats.columns = [
        'category_price_mean', 'category_price_median', 'category_product_count',
        'category_sales_mean', 'category_sales_median', 'category_sales_total',
        'category_quality_mean', 'category_quality_median', 'category_quality_min', 'category_quality_max'
    ]
    
    # Reset index
    category_stats = category_stats.reset_index()
    
    # Step 2: Identify categories with high sales potential
    print("2. Identifying high-potential categories...")
    
    # Calculate thresholds for high-potential categories
    price_threshold = category_stats['category_price_mean'].quantile(0.75)  # Top 25% by price
    sales_threshold = category_stats['category_sales_total'].quantile(0.75)  # Top 25% by sales
    
    high_potential_categories = category_stats[
        (category_stats['category_price_mean'] >= price_threshold) &
        (category_stats['category_sales_total'] >= sales_threshold)
    ].copy()
    
    high_potential_categories['is_high_potential'] = True
    
    print(f"  Found {len(high_potential_categories)} high-potential categories")
    print(f"  Price threshold: ${price_threshold:.2f}")
    print(f"  Sales threshold: {sales_threshold:.0f} total sales")
    
    # Step 3: Merge category stats back to main dataframe
    df = df.merge(
        category_stats[[category_col, 'category_price_mean', 'category_sales_total']],
        on=category_col,
        how='left'
    )
    
    # Step 4: Calculate quality score percentiles within each category
    print("3. Calculating quality score percentiles within categories...")
    
    def calculate_quality_percentile(group):
        """Calculate percentile rank within category"""
        if len(group) >= 4:  # Need at least 4 products for meaningful percentiles
            return group[quality_score_col].rank(pct=True) * 100
        else:
            return pd.Series([50] * len(group), index=group.index)
    
    df['quality_percentile_in_category'] = df.groupby(category_col)[quality_score_col].transform(
        calculate_quality_percentile
    )
    
    # Step 5: Filter for low-quality products
    print("4. Identifying low-quality products...")
    
    # Products in bottom 25th percentile of quality within their category
    low_quality_mask = df['quality_percentile_in_category'] <= 25
    
    print(f"  Found {low_quality_mask.sum()} products in bottom 25th percentile of quality")
    
    # Step 6: Combine criteria for high-priority listings
    print("5. Applying combined criteria for high-priority listings...")
    
    # Criteria:
    # 1. Low quality (bottom 25th percentile in category)
    # 2. In high-potential category (or at least decent category metrics)
    # 3. Individual product has decent sales (not bottom performers)
    
    # Calculate sales percentile within category
    df['sales_percentile_in_category'] = df.groupby(category_col)[sales_col].transform(
        lambda x: x.rank(pct=True) * 100 if len(x) >= 4 else 50
    )
    
    # Define final selection criteria
    high_priority_mask = (
        low_quality_mask &  # Low quality
        (df['category_price_mean'] >= price_threshold * 0.5) &  # At least half of high threshold
        (df['category_sales_total'] >= sales_threshold * 0.5) &  # At least half of high threshold
        (df['sales_percentile_in_category'] >= 50)  # At least median sales in category
    )
    
    # Get high-priority listings
    high_priority_df = df[high_priority_mask].copy()
    
    # Step 7: Calculate priority score
    print("6. Calculating priority scores...")
    
    def calculate_priority_score(row, max_price, max_sales, max_quality_gap):
        """
        Calculate priority score (higher = more urgent)
        
        Factors:
        1. How much sales could improve if quality improves (sales percentile)
        2. How much room for quality improvement (quality gap)
        3. Category importance (price and total sales)
        """
        
        # Normalize factors
        sales_factor = row['sales_percentile_in_category'] / 100  # Higher sales = more potential
        quality_gap_factor = (100 - row[quality_score_col]) / 100  # Larger gap = more potential
        category_price_factor = row['category_price_mean'] / max_price if max_price > 0 else 0
        category_sales_factor = row['category_sales_total'] / max_sales if max_sales > 0 else 0
        
        # Calculate priority score (0-100)
        priority_score = (
            (sales_factor * 30) +  # Sales potential weight
            (quality_gap_factor * 40) +  # Quality improvement weight
            (category_price_factor * 15) +  # Category price weight
            (category_sales_factor * 15)  # Category sales weight
        )
        
        return min(priority_score, 100)
    
    # Get max values for normalization
    max_price = df['category_price_mean'].max()
    max_sales = df['category_sales_total'].max()
    max_quality_gap = (100 - df[quality_score_col]).max()
    
    # Apply priority score calculation
    high_priority_df['priority_score'] = high_priority_df.apply(
        lambda row: calculate_priority_score(row, max_price, max_sales, max_quality_gap),
        axis=1
    )
    
    # Sort by priority score (highest first)
    high_priority_df = high_priority_df.sort_values('priority_score', ascending=False)
    
    # Step 8: Create analysis summary
    print("7. Creating analysis summary...")
    
    summary = {
        'total_products_analyzed': len(df),
        'high_priority_count': len(high_priority_df),
        'high_priority_percentage': (len(high_priority_df) / len(df)) * 100,
        'category_stats': {
            'total_categories': df[category_col].nunique(),
            'high_potential_categories': len(high_potential_categories),
            'price_threshold': price_threshold,
            'sales_threshold': sales_threshold
        },
        'quality_stats': {
            'avg_quality_score_all': df[quality_score_col].mean(),
            'avg_quality_score_high_priority': high_priority_df[quality_score_col].mean() if len(high_priority_df) > 0 else 0,
            'quality_cutoff_percentile': 25
        },
        'sales_stats': {
            'avg_sales_all': df[sales_col].mean(),
            'avg_sales_high_priority': high_priority_df[sales_col].mean() if len(high_priority_df) > 0 else 0,
            'min_sales_percentile_required': 50
        }
    }
    
    # Step 9: Display results
    print("\n" + "="*80)
    print("📊 HIGH-PRIORITY LISTINGS ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📈 OVERVIEW:")
    print(f"  Total products analyzed: {summary['total_products_analyzed']:,}")
    print(f"  High-priority listings identified: {summary['high_priority_count']:,}")
    print(f"  Percentage of total: {summary['high_priority_percentage']:.1f}%")
    
    print(f"\n🏷️ CATEGORY FILTERS:")
    print(f"  High-potential categories: {summary['category_stats']['high_potential_categories']}")
    print(f"  Price threshold: ${summary['category_stats']['price_threshold']:.2f}")
    print(f"  Sales threshold: {summary['category_stats']['sales_threshold']:,.0f} total sales")
    
    print(f"\n📊 QUALITY FILTERS:")
    print(f"  Quality cutoff: Bottom {summary['quality_stats']['quality_cutoff_percentile']}th percentile")
    print(f"  Avg quality (all): {summary['quality_stats']['avg_quality_score_all']:.1f}")
    print(f"  Avg quality (high-priority): {summary['quality_stats']['avg_quality_score_high_priority']:.1f}")
    
    print(f"\n💰 SALES FILTERS:")
    print(f"  Min sales percentile: {summary['sales_stats']['min_sales_percentile_required']}th percentile")
    print(f"  Avg sales (all): {summary['sales_stats']['avg_sales_all']:.1f}")
    print(f"  Avg sales (high-priority): {summary['sales_stats']['avg_sales_high_priority']:.1f}")
    
    if len(high_priority_df) > 0:
        print(f"\n🏆 TOP 10 HIGHEST PRIORITY LISTINGS:")
        print("-"*80)
        print(f"{'Rank':<5} {'Product ID':<12} {'Priority':<10} {'Quality':<8} {'Sales':<10} {'Price':<10} {'Category':<15}")
        print("-"*80)
        
        for i, (_, row) in enumerate(high_priority_df.head(10).iterrows(), 1):
            print(f"{i:<5} {str(row[id_col]):<12} {row['priority_score']:<10.1f} "
                  f"{row[quality_score_col]:<8.1f} {row[sales_col]:<10.0f} "
                  f"${row[price_col]:<9.2f} {str(row[category_col]):<15}")
        
        print(f"\n📋 HIGH-PRIORITY LISTINGS BY CATEGORY:")
        category_counts = high_priority_df[category_col].value_counts()
        for category, count in category_counts.head(10).items():
            percentage = (count / len(high_priority_df)) * 100
            print(f"  {str(category):<20}: {count:>3} listings ({percentage:>5.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    return high_priority_df, summary


def find_low_quality_high_potential_listings(scored_df):
    """
    Find listings where:
    1. quality_score is in bottom 25th percentile within its category_id
    2. The category has at least 10 products
    
    Parameters:
    -----------
    scored_df : pandas DataFrame
        DataFrame containing at least: id, category_id, quality_score, sold_quantity, price
        
    Returns:
    --------
    pandas DataFrame
        Filtered DataFrame with low-quality listings in high-potential categories
    dict
        Analysis summary
    """
    
    print("🔍 FINDING LOW-QUALITY LISTINGS IN HIGH-POTENTIAL CATEGORIES")
    print("="*80)
    
    # Create a working copy
    df = scored_df.copy()
    
    # Step 1: Filter categories with at least 10 products
    print("1. Filtering categories with at least 10 products...")
    
    # Count products in each category
    category_counts = df['category_id'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index.tolist()
    
    # Filter DataFrame
    df_filtered = df[df['category_id'].isin(valid_categories)].copy()
    
    print(f"   Total categories: {df['category_id'].nunique()}")
    print(f"   Categories with >=10 products: {len(valid_categories)}")
    print(f"   Products in valid categories: {len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        print("❌ No categories with at least 10 products found!")
        return pd.DataFrame(), {}
    
    # Step 2: Calculate quality score percentiles within each category
    print("\n2. Calculating quality score percentiles within each category...")
    
    def calculate_percentile(group):
        """Calculate percentile rank for quality_score within category"""
        return group.rank(pct=True) * 100
    
    df_filtered['quality_percentile'] = df_filtered.groupby('category_id')['quality_score'].transform(
        calculate_percentile
    )
    
    # Step 3: Identify listings in bottom 25th percentile
    print("\n3. Identifying listings in bottom 25th percentile...")
    
    low_quality_mask = df_filtered['quality_percentile'] <= 25
    low_quality_df = df_filtered[low_quality_mask].copy()
    
    print(f"   Low-quality listings found: {len(low_quality_df):,}")
    print(f"   Percentage of valid listings: {(len(low_quality_df)/len(df_filtered))*100:.1f}%")
    
    # Step 4: Calculate category statistics for context
    print("\n4. Calculating category statistics...")
    
    category_stats = df_filtered.groupby('category_id').agg({
        'id': 'count',
        'quality_score': ['mean', 'min', 'max'],
        'sold_quantity': ['mean', 'sum','max'],
        'price': ['mean', 'median']
    }).round(2)
    
    # Flatten column names
    category_stats.columns = [
        'product_count',
        'avg_quality', 'min_quality', 'max_quality',
        'avg_sales', 'total_sales','max_sales',
        'avg_price', 'median_price'
    ]
    
    category_stats = category_stats.reset_index()
    
    # Step 5: Merge category stats with low-quality listings
    print("\n5. Enriching low-quality listings with category stats...")
    
    result_df = low_quality_df.merge(
        category_stats,
        on='category_id',
        how='left',
        suffixes=('', '_category')
    )
    
    # Step 6: Calculate improvement priority score
    print("\n6. Calculating improvement priority score...")
    
    def calculate_priority_score(row):
        """
        Calculate priority score based on:
        1. How low the quality score is
        2. Category sales performance
        3. Category price level
        """
        # Quality gap factor (lower quality = higher priority)
        quality_gap = 100 - row['quality_score']
        quality_factor = quality_gap / 100
        
        # Check if we have category stats
        if 'max_sales' in row: 
            # Sales factor (higher sales = higher priority)
            sales_factor = min(row['max_sales'] / 1000, 1.0)  # Normalize
        else:
            sales_factor = 0.5  # Default value
        
        if 'median_price' in row:  
            # Price factor (higher price = higher priority)
            price_factor = min(row['median_price'] / 500, 1.0)  # Normalize
        else:
            price_factor = 0.5  # Default value
        
        # Combined priority score (0-100)
        priority_score = (
            quality_factor * 50 +    # 50% weight on quality gap
            sales_factor * 30 +      # 30% weight on category sales
            price_factor * 20        # 20% weight on category price
        )
        
        return min(priority_score, 100)
    
    if len(result_df) > 0:
        
        # Check which columns are available
        required_cols = ['quality_score', 'max_sales', 'median_price']
        available_cols = [col for col in required_cols if col in result_df.columns]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            print(f"   Warning: Missing columns: {missing_cols}")
            print(f"   Using default values for missing columns")
        
        result_df['priority_score'] = result_df.apply(calculate_priority_score, axis=1)
        
        # Sort by priority score
        result_df = result_df.sort_values('priority_score', ascending=False)
    
    # Step 7: Create analysis summary
    print("\n7. Creating analysis summary...")
    
    summary = {
        'overall': {
            'total_listings': len(scored_df),
            'listings_in_valid_categories': len(df_filtered),
            'low_quality_listings_found': len(result_df),
            'percentage_low_quality': (len(result_df) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        },
        'category_summary': {
            'total_categories': df['category_id'].nunique(),
            'categories_with_10plus': len(valid_categories),
            'avg_products_per_category': len(df_filtered) / len(valid_categories) if len(valid_categories) > 0 else 0
        },
        'quality_stats': {
            'overall_avg_quality': scored_df['quality_score'].mean(),
            'filtered_avg_quality': df_filtered['quality_score'].mean(),
            'low_quality_avg': result_df['quality_score'].mean() if len(result_df) > 0 else 0,
            'quality_cutoff_percentile': 25
        },
        'sales_stats': {
            'overall_avg_sales': scored_df['sold_quantity'].mean(),
            'filtered_avg_sales': df_filtered['sold_quantity'].mean(),
            'low_quality_avg_sales': result_df['sold_quantity'].mean() if len(result_df) > 0 else 0
        }
    }
    
    # Step 8: Display results
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📈 OVERVIEW:")
    print(f"  Total listings analyzed: {summary['overall']['total_listings']:,}")
    print(f"  Listings in categories with >=10 products: {summary['overall']['listings_in_valid_categories']:,}")
    print(f"  Low-quality listings found: {summary['overall']['low_quality_listings_found']:,}")
    print(f"  Percentage: {summary['overall']['percentage_low_quality']:.1f}%")
    
    print(f"\n🏷️ CATEGORY STATS:")
    print(f"  Total categories: {summary['category_summary']['total_categories']}")
    print(f"  Categories with >=10 products: {summary['category_summary']['categories_with_10plus']}")
    print(f"  Avg products per category: {summary['category_summary']['avg_products_per_category']:.1f}")
    
    print(f"\n📊 QUALITY STATS:")
    print(f"  Overall avg quality score: {summary['quality_stats']['overall_avg_quality']:.1f}")
    print(f"  Filtered avg quality score: {summary['quality_stats']['filtered_avg_quality']:.1f}")
    print(f"  Low-quality listings avg: {summary['quality_stats']['low_quality_avg']:.1f}")
    print(f"  Quality cutoff: Bottom {summary['quality_stats']['quality_cutoff_percentile']}th percentile")
    
    if len(result_df) > 0:
        print(f"\n💰 SALES & PRICE STATS:")
        print(f"  Overall avg sales: {summary['sales_stats']['overall_avg_sales']:.1f}")
        print(f"  Filtered avg sales: {summary['sales_stats']['filtered_avg_sales']:.1f}")
        print(f"  Low-quality listings avg sales: {summary['sales_stats']['low_quality_avg_sales']:.1f}")
        
        print(f"\n🏆 TOP 10 HIGHEST PRIORITY LISTINGS:")
        print("-"*100)
        print(f"{'Rank':<5} {'ID':<10} {'Category':<15} {'Quality':<10} {'Sales':<10} {'Max Sales':<10} {'Price':<10} {'Priority':<10}")
        print("-"*100)
        
        top_10 = result_df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:<5} {row['id']:<10} {row['category_id']:<15} "
                  f"{row['quality_score']:<10.1f} {row['sold_quantity']:<10.0f} {row['max_sales']:<10.0f}"
                  f"${row['price']:<9.2f} {row['priority_score']:<10.1f}")
        
        print(f"\n📊 DISTRIBUTION BY CATEGORY:")
        category_dist = result_df['category_id'].value_counts().head(10)
        for category, count in category_dist.items():
            percentage = (count / len(result_df)) * 100
            print(f"  {category:<20}: {count:>3} listings ({percentage:>5.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    return result_df, summary


def find_title_problem_listings(scored_df):
    """
    Find listings where:
    1. quality_score is between 25th and 50th percentile within its category_id
    2. The category has at least 10 products
    3. The main problem is title length or title quality
    4. Sort by category's max sales and median price
    
    Parameters:
    -----------
    scored_df : pandas DataFrame
        DataFrame containing at least: id, category_id, quality_score, sold_quantity, price, 
        title_length, title_score, score_breakdown
        
    Returns:
    --------
    pandas DataFrame
        Filtered DataFrame with listings having title-related problems
    dict
        Analysis summary
    """
    
    print("🔍 FINDING LISTINGS WITH TITLE-RELATED PROBLEMS")
    print("="*80)
    
    # Create a working copy
    df = scored_df.copy()
    
    # Step 1: Filter categories with at least 10 products
    print("1. Filtering categories with at least 10 products...")
    
    # Count products in each category
    category_counts = df['category_id'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index.tolist()
    
    # Filter DataFrame
    df_filtered = df[df['category_id'].isin(valid_categories)].copy()
    
    print(f"   Total categories: {df['category_id'].nunique()}")
    print(f"   Categories with >=10 products: {len(valid_categories)}")
    print(f"   Products in valid categories: {len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        print("❌ No categories with at least 10 products found!")
        return pd.DataFrame(), {}
    
    # Step 2: Calculate quality score percentiles within each category
    print("\n2. Calculating quality score percentiles within each category...")
    
    def calculate_percentile(group):
        """Calculate percentile rank for quality_score within category"""
        return group.rank(pct=True) * 100
    
    df_filtered['quality_percentile'] = df_filtered.groupby('category_id')['quality_score'].transform(
        calculate_percentile
    )
    
    # Step 3: Filter listings in 25th to 50th percentile
    print("\n3. Filtering listings in 25th to 50th percentile...")
    
    mid_quality_mask = (df_filtered['quality_percentile'] >= 25) & (df_filtered['quality_percentile'] <= 50)
    mid_quality_df = df_filtered[mid_quality_mask].copy()
    
    print(f"   Mid-quality listings found: {len(mid_quality_df):,}")
    print(f"   Percentage of valid listings: {(len(mid_quality_df)/len(df_filtered))*100:.1f}%")
    
    if len(mid_quality_df) == 0:
        print("❌ No mid-quality listings found!")
        return pd.DataFrame(), {}
    
    # Step 4: Check for title-related problems
    print("\n4. Identifying listings with title-related problems...")
    
    def has_title_problem(row):
        """Check if listing has title length or title quality problems"""
        title_problems = []
        
        # Check title_length if available
        if 'title_length' in row and pd.notna(row['title_length']):
            if row['title_length'] < 30:  # Too short
                title_problems.append('Title too short')
            elif row['title_length'] > 100:  # Too long (optional)
                title_problems.append('Title too long')
        
        # Check title_score if available
        if 'title_score' in row and pd.notna(row['title_score']):
            if row['title_score'] < 60:  # Low title quality
                title_problems.append('Low title quality')
        
        # Check score_breakdown for title-related issues
        if 'score_breakdown' in row and isinstance(row['score_breakdown'], dict):
            breakdown = row['score_breakdown']
            for key, data in breakdown.items():
                if any(title_word in key.lower() for title_word in ['title', 'name', 'heading']):
                    if isinstance(data, dict) and 'raw_score' in data:
                        if data['raw_score'] < 60:
                            title_problems.append(f'Low {key} score')
        
        return len(title_problems) > 0, title_problems
    
    # Apply title problem detection
    title_problem_results = mid_quality_df.apply(has_title_problem, axis=1, result_type='expand')
    mid_quality_df['has_title_problem'] = title_problem_results[0]
    mid_quality_df['title_problems'] = title_problem_results[1]
    
    # Filter for listings with title problems
    title_problem_df = mid_quality_df[mid_quality_df['has_title_problem']].copy()
    
    print(f"   Listings with title problems: {len(title_problem_df):,}")
    print(f"   Percentage of mid-quality listings: {(len(title_problem_df)/len(mid_quality_df))*100:.1f}%")
    
    if len(title_problem_df) == 0:
        print("❌ No listings with title problems found!")
        return pd.DataFrame(), {}
    
    # Step 5: Calculate category statistics for sorting
    print("\n5. Calculating category statistics for sorting...")
    
    category_stats = df_filtered.groupby('category_id').agg({
        'sold_quantity': ['max', 'mean', 'median'],
        'price': ['median', 'mean'],
        'id': 'count'
    }).round(2)
    
    # Flatten column names
    category_stats.columns = [
        'max_sales', 'avg_sales', 'median_sales',
        'median_price', 'avg_price',
        'product_count'
    ]
    
    category_stats = category_stats.reset_index()
    
    # Step 6: Merge category stats
    print("\n6. Merging category statistics...")
    
    result_df = title_problem_df.merge(
        category_stats,
        on='category_id',
        how='left',
        suffixes=('', '_category')
    )
    
    # Step 7: Sort by category max sales and median price
    print("\n7. Sorting by category performance...")
    
    # First, sort categories by max_sales (descending) and median_price (descending)
    # Create a category ranking
    category_ranking = category_stats.sort_values(
        ['max_sales', 'median_price'], 
        ascending=[False, False]
    )
    category_ranking['category_rank'] = range(1, len(category_ranking) + 1)
    
    # Merge ranking back to result_df
    result_df = result_df.merge(
        category_ranking[['category_id', 'category_rank']],
        on='category_id',
        how='left'
    )
    
    # Sort results: first by category_rank, then by severity of title problems
    def calculate_title_problem_severity(problems):
        """Calculate severity score based on number and type of problems"""
        severity = 0
        if isinstance(problems, list):
            severity += len(problems) * 10  # Each problem adds 10 points
            
            # Add extra points for specific serious problems
            for problem in problems:
                if 'too short' in problem.lower():
                    severity += 5
                elif 'low quality' in problem.lower():
                    severity += 3
                elif 'low score' in problem.lower():
                    severity += 2
        
        return severity
    
    result_df['title_problem_severity'] = result_df['title_problems'].apply(calculate_title_problem_severity)
    
    # Final sorting
    result_df = result_df.sort_values(
        ['category_rank', 'title_problem_severity', 'quality_score'],
        ascending=[True, False, True]  # Lower quality_score = more severe
    )
    
    # Step 8: Create analysis summary
    print("\n8. Creating analysis summary...")
    
    summary = {
        'overall': {
            'total_listings': len(df),
            'listings_in_valid_categories': len(df_filtered),
            'mid_quality_listings': len(mid_quality_df),
            'title_problem_listings': len(result_df),
            'percentage_title_problems': (len(result_df) / len(mid_quality_df) * 100) if len(mid_quality_df) > 0 else 0
        },
        'category_summary': {
            'total_categories': df['category_id'].nunique(),
            'categories_with_10plus': len(valid_categories),
            'categories_with_title_problems': result_df['category_id'].nunique() if len(result_df) > 0 else 0
        },
        'quality_stats': {
            'overall_avg_quality': df['quality_score'].mean(),
            'mid_quality_avg': mid_quality_df['quality_score'].mean() if len(mid_quality_df) > 0 else 0,
            'title_problem_avg': result_df['quality_score'].mean() if len(result_df) > 0 else 0,
            'quality_range': '25th-50th percentile'
        },
        'title_problem_stats': {
            'most_common_problem': None,
            'avg_problems_per_listing': result_df['title_problem_severity'].mean() / 10 if len(result_df) > 0 else 0
        }
    }
    
    # Calculate most common title problem
    if len(result_df) > 0:
        all_problems = []
        for problems in result_df['title_problems']:
            if isinstance(problems, list):
                all_problems.extend(problems)
        
        if all_problems:
            from collections import Counter
            problem_counts = Counter(all_problems)
            most_common = problem_counts.most_common(1)
            if most_common:
                summary['title_problem_stats']['most_common_problem'] = most_common[0][0]
    
    # Step 9: Display results
    print("\n" + "="*80)
    print("📊 TITLE PROBLEM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📈 OVERVIEW:")
    print(f"  Total listings analyzed: {summary['overall']['total_listings']:,}")
    print(f"  Listings in categories with >=10 products: {summary['overall']['listings_in_valid_categories']:,}")
    print(f"  Mid-quality listings (25th-50th %ile): {summary['overall']['mid_quality_listings']:,}")
    print(f"  Listings with title problems: {summary['overall']['title_problem_listings']:,}")
    print(f"  Percentage with title problems: {summary['overall']['percentage_title_problems']:.1f}%")
    
    print(f"\n🏷️ CATEGORY STATS:")
    print(f"  Total categories: {summary['category_summary']['total_categories']}")
    print(f"  Categories with >=10 products: {summary['category_summary']['categories_with_10plus']}")
    print(f"  Categories with title problems: {summary['category_summary']['categories_with_title_problems']}")
    
    print(f"\n📊 QUALITY STATS:")
    print(f"  Overall avg quality: {summary['quality_stats']['overall_avg_quality']:.1f}")
    print(f"  Mid-quality avg: {summary['quality_stats']['mid_quality_avg']:.1f}")
    print(f"  Title problem listings avg: {summary['quality_stats']['title_problem_avg']:.1f}")
    print(f"  Quality range: {summary['quality_stats']['quality_range']}")
    
    if summary['title_problem_stats']['most_common_problem']:
        print(f"\n📝 TITLE PROBLEM STATS:")
        print(f"  Most common problem: {summary['title_problem_stats']['most_common_problem']}")
        print(f"  Avg problems per listing: {summary['title_problem_stats']['avg_problems_per_listing']:.1f}")
    
    if len(result_df) > 0:
        print(f"\n🏆 TOP 10 LISTINGS WITH TITLE PROBLEMS (Sorted by Category Performance):")
        print("-"*120)
        print(f"{'Rank':<5} {'ID':<10} {'Category':<15} {'Quality':<10} {'Sales':<10} {'Title Score':<12} {'Title Len':<10} {'Problems':<30}")
        print("-"*120)
        
        top_10 = result_df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            # Get title score if available
            title_score = row.get('title_score', 'N/A')
            if title_score != 'N/A':
                title_score = f"{title_score:.1f}"
            
            # Get title length if available
            title_len = row.get('title_length', 'N/A')
            if title_len != 'N/A':
                title_len = f"{int(title_len)}"
            
            # Get problems as string
            problems = row.get('title_problems', [])
            if isinstance(problems, list):
                problems_str = ', '.join(problems[:2])  # Show first 2 problems
                if len(problems) > 2:
                    problems_str += f"... (+{len(problems)-2})"
            else:
                problems_str = str(problems)[:30]
            
            print(f"{i:<5} {row['id']:<10} {row['category_id']:<15} "
                  f"{row['quality_score']:<10.1f} {row['sold_quantity']:<10.0f} "
                  f"{title_score:<12} {title_len:<10} {problems_str:<30}")
        
        print(f"\n📊 CATEGORIES WITH MOST TITLE PROBLEMS:")
        category_problem_counts = result_df['category_id'].value_counts().head(5)
        for category, count in category_problem_counts.items():
            category_stats_row = category_stats[category_stats['category_id'] == category].iloc[0]
            print(f"  {category:<20}: {count:>3} listings | Max sales: {category_stats_row['max_sales']:,.0f} | "
                  f"Median price: ${category_stats_row['median_price']:.2f}")
    
    print("\n" + "="*80)
    print("✅ TITLE PROBLEM ANALYSIS COMPLETE")
    print("="*80)
    
    return result_df, summary


import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def analyze_quality_improvement_impact(
    scored_df: pd.DataFrame,
    quality_score_col: str = 'quality_score',
    sales_rank_col: str = 'sales_rank',
    sold_quantity_col: str = 'sold_quantity',
    category_id_col: str = 'category_id',
    price_col: str = 'price',
    correlation_effect_multiplier: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the impact of improving quality scores for products below the 50th percentile.
    
    Parameters:
    -----------
    scored_df : pd.DataFrame
        DataFrame containing quality scores and sales data
    quality_score_col : str
        Column name for quality score
    sales_rank_col : str
        Column name for sales rank within category
    sold_quantity_col : str
        Column name for sold quantity
    category_id_col : str
        Column name for category ID
    price_col : str
        Column name for price
    correlation_effect_multiplier : float
        Multiplier for correlation effect (default 0.5 = conservative estimate)
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results with improvement table
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df = scored_df.copy()
    
    # Initialize results dictionary
    results = {
        'correlation_analysis': {},
        'improvement_table': pd.DataFrame(),
        'summary_statistics': {}
    }
    
    # 1. Calculate correlation between quality_score and sales_rank
    print("1. Calculating correlation between quality_score and sales_rank...")
    
    # Check if required columns exist
    required_cols = [quality_score_col, sales_rank_col, sold_quantity_col, category_id_col, price_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data - remove rows with missing values in key columns
    df_clean = df.dropna(subset=[quality_score_col, sales_rank_col, sold_quantity_col, category_id_col])
    
    # Calculate Pearson correlation coefficient
    # Note: sales_rank is reversed (lower rank = better performance)
    # We expect negative correlation with quality_score
    corr_coefficient, p_value = stats.pearsonr(
        df_clean[quality_score_col], 
        df_clean[sales_rank_col]
    )
    
    # Calculate Spearman correlation (non-parametric, robust to outliers)
    spearman_corr, spearman_p = stats.spearmanr(
        df_clean[quality_score_col], 
        df_clean[sales_rank_col]
    )
    
    # Store correlation results
    results['correlation_analysis'] = {
        'pearson': {
            'coefficient': corr_coefficient,
            'p_value': p_value,
            'interpretation': _interpret_correlation(corr_coefficient, p_value),
            'is_significant': p_value < 0.05
        },
        'spearman': {
            'coefficient': spearman_corr,
            'p_value': spearman_p,
            'interpretation': _interpret_correlation(spearman_corr, spearman_p),
            'is_significant': spearman_p < 0.05
        },
        'n_samples': len(df_clean),
        'correlation_effect_multiplier': correlation_effect_multiplier
    }
    
    print(f"   Pearson correlation: {corr_coefficient:.4f} (p-value: {p_value:.6f})")
    print(f"   Interpretation: {results['correlation_analysis']['pearson']['interpretation']}")
    
    # 2. Identify products below 50th percentile in their category
    print("\n2. Identifying low-quality products (below 50th percentile)...")
    
    # Calculate category-wise percentiles
    df_clean['quality_percentile'] = df_clean.groupby(category_id_col)[quality_score_col].transform(
        lambda x: x.rank(pct=True) * 100
    )
    
    # Identify products below 50th percentile in their category
    low_quality_mask = df_clean['quality_percentile'] < 50
    low_quality_df = df_clean[low_quality_mask].copy()
    high_quality_df = df_clean[~low_quality_mask].copy()
    
    # Store summary statistics
    results['summary_statistics'] = {
        'total_products': len(df_clean),
        'low_quality_products': len(low_quality_df),
        'high_quality_products': len(high_quality_df),
        'low_quality_percentage': (len(low_quality_df) / len(df_clean)) * 100,
        'current_total_sales': df_clean[sold_quantity_col].sum(),
        'current_total_revenue': (df_clean[sold_quantity_col] * df_clean[price_col]).sum(),
        'avg_quality_score_low': low_quality_df[quality_score_col].mean(),
        'avg_quality_score_high': high_quality_df[quality_score_col].mean(),
        'avg_sales_low': low_quality_df[sold_quantity_col].mean(),
        'avg_sales_high': high_quality_df[sold_quantity_col].mean(),
        'avg_revenue_low': (low_quality_df[sold_quantity_col] * low_quality_df[price_col]).mean(),
        'avg_revenue_high': (high_quality_df[sold_quantity_col] * high_quality_df[price_col]).mean()
    }
    
    print(f"   Total products analyzed: {len(df_clean):,}")
    print(f"   Low-quality products (below 50th percentile): {len(low_quality_df):,} ({results['summary_statistics']['low_quality_percentage']:.1f}%)")
    print(f"   Current total sales: {results['summary_statistics']['current_total_sales']:,.0f}")
    print(f"   Current total revenue: ${results['summary_statistics']['current_total_revenue']:,.2f}")
    
    # 3. Calculate improvement impact for different quality improvement levels
    print("\n3. Calculating impact of quality improvement for low-quality products...")
    
    # Define quality improvement percentages to analyze
    improvement_levels = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5%, 10%, 15%, 20%, 25%
    
    # Calculate current totals for low-quality products
    current_low_quality_sales = low_quality_df[sold_quantity_col].sum()
    current_low_quality_revenue = (low_quality_df[sold_quantity_col] * low_quality_df[price_col]).sum()
    
    # Prepare improvement analysis table
    improvement_data = []
    
    for improvement_pct in improvement_levels:
        # Calculate uplift factor based on correlation
        # Formula: uplift = 1 + (abs(correlation) × improvement_pct × multiplier)
        uplift_factor = 1 + (abs(corr_coefficient) * improvement_pct * correlation_effect_multiplier)
        
        # Calculate new sales quantity for low-quality products
        new_low_quality_sales = current_low_quality_sales * uplift_factor
        
        # Calculate new revenue for low-quality products
        new_low_quality_revenue = current_low_quality_revenue * uplift_factor
        
        # Calculate improvements
        sales_improvement = new_low_quality_sales - current_low_quality_sales
        revenue_improvement = new_low_quality_revenue - current_low_quality_revenue
        
        # Calculate percentage improvements
        sales_improvement_pct = (sales_improvement / current_low_quality_sales) * 100 if current_low_quality_sales > 0 else 0
        revenue_improvement_pct = (revenue_improvement / current_low_quality_revenue) * 100 if current_low_quality_revenue > 0 else 0
        
        # Calculate average improvement per product
        avg_sales_improvement_per_product = sales_improvement / len(low_quality_df) if len(low_quality_df) > 0 else 0
        avg_revenue_improvement_per_product = revenue_improvement / len(low_quality_df) if len(low_quality_df) > 0 else 0
        
        # Store results for this improvement level
        improvement_data.append({
            'quality_improvement_pct': improvement_pct * 100,  # Convert to percentage
            'quality_improvement_label': f"{improvement_pct*100:.0f}%",
            'uplift_factor': uplift_factor,
            'current_low_quality_sales': current_low_quality_sales,
            'current_low_quality_revenue': current_low_quality_revenue,
            'new_low_quality_sales': new_low_quality_sales,
            'new_low_quality_revenue': new_low_quality_revenue,
            'sales_improvement': sales_improvement,
            'revenue_improvement': revenue_improvement,
            'sales_improvement_pct': sales_improvement_pct,
            'revenue_improvement_pct': revenue_improvement_pct,
            'avg_sales_improvement_per_product': avg_sales_improvement_per_product,
            'avg_revenue_improvement_per_product': avg_revenue_improvement_per_product,
            'affected_products': len(low_quality_df)
        })
    
    # Create improvement table DataFrame
    improvement_df = pd.DataFrame(improvement_data)
    
    # Format the table for display
    display_columns = [
        'quality_improvement_label',
        'sales_improvement',
        'sales_improvement_pct',
        'revenue_improvement',
        'revenue_improvement_pct',
        'avg_sales_improvement_per_product',
        'avg_revenue_improvement_per_product',
        'uplift_factor',
        'affected_products'
    ]
    
    display_df = improvement_df[display_columns].copy()
    
    # Rename columns for better readability
    column_rename_map = {
        'quality_improvement_label': 'Quality Improvement',
        'sales_improvement': 'Sales Improvement (Units)',
        'sales_improvement_pct': 'Sales Improvement (%)',
        'revenue_improvement': 'Revenue Improvement ($)',
        'revenue_improvement_pct': 'Revenue Improvement (%)',
        'avg_sales_improvement_per_product': 'Avg Sales/Product',
        'avg_revenue_improvement_per_product': 'Avg Revenue/Product ($)',
        'uplift_factor': 'Uplift Factor',
        'affected_products': 'Affected Products'
    }
    
    display_df = display_df.rename(columns=column_rename_map)
    
    # Format numeric columns
    display_df['Sales Improvement (Units)'] = display_df['Sales Improvement (Units)'].apply(lambda x: f"{x:,.0f}")
    display_df['Revenue Improvement ($)'] = display_df['Revenue Improvement ($)'].apply(lambda x: f"${x:,.2f}")
    display_df['Sales Improvement (%)'] = display_df['Sales Improvement (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Revenue Improvement (%)'] = display_df['Revenue Improvement (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Avg Sales/Product'] = display_df['Avg Sales/Product'].apply(lambda x: f"{x:.2f}")
    display_df['Avg Revenue/Product ($)'] = display_df['Avg Revenue/Product ($)'].apply(lambda x: f"${x:.2f}")
    display_df['Uplift Factor'] = display_df['Uplift Factor'].apply(lambda x: f"{x:.4f}")
    display_df['Affected Products'] = display_df['Affected Products'].apply(lambda x: f"{x:,}")
    
    # Store both raw and formatted tables in results
    results['improvement_table_raw'] = improvement_df
    results['improvement_table_formatted'] = display_df
    results['low_quality_data'] = {
        'current_sales': current_low_quality_sales,
        'current_revenue': current_low_quality_revenue,
        'product_count': len(low_quality_df)
    }
    
    # Print the improvement table
    print("\n" + "=" * 100)
    print("QUALITY IMPROVEMENT IMPACT TABLE")
    print("=" * 100)
    print("Impact of improving quality scores for products below 50th percentile")
    print(f"Affected products: {len(low_quality_df):,} (Current sales: {current_low_quality_sales:,.0f} units, Revenue: ${current_low_quality_revenue:,.2f})")
    print(f"Correlation effect multiplier: {correlation_effect_multiplier}")
    print("-" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)
    
    # Print summary insights
    print("\n4. KEY INSIGHTS")
    print("=" * 60)
    
    # Get the 10% improvement case for summary
    ten_percent_case = improvement_df[improvement_df['quality_improvement_pct'] == 10].iloc[0]
    
    print(f"• With 10% quality improvement for low-quality products:")
    print(f"  → Sales increase: {ten_percent_case['sales_improvement']:,.0f} units ({ten_percent_case['sales_improvement_pct']:.2f}%)")
    print(f"  → Revenue increase: ${ten_percent_case['revenue_improvement']:,.2f} ({ten_percent_case['revenue_improvement_pct']:.2f}%)")
    print(f"  → Average per product: {ten_percent_case['avg_sales_improvement_per_product']:.1f} units, ${ten_percent_case['avg_revenue_improvement_per_product']:.2f}")
    
    # Calculate potential at 25% improvement
    twenty_five_percent_case = improvement_df[improvement_df['quality_improvement_pct'] == 25].iloc[0]
    
    print(f"\n• With 25% quality improvement for low-quality products:")
    print(f"  → Sales increase: {twenty_five_percent_case['sales_improvement']:,.0f} units ({twenty_five_percent_case['sales_improvement_pct']:.2f}%)")
    print(f"  → Revenue increase: ${twenty_five_percent_case['revenue_improvement']:,.2f} ({twenty_five_percent_case['revenue_improvement_pct']:.2f}%)")
    
    # Calculate total portfolio impact
    total_sales = results['summary_statistics']['current_total_sales']
    total_revenue = results['summary_statistics']['current_total_revenue']
    
    portfolio_sales_impact_pct = (ten_percent_case['sales_improvement'] / total_sales) * 100 if total_sales > 0 else 0
    portfolio_revenue_impact_pct = (ten_percent_case['revenue_improvement'] / total_revenue) * 100 if total_revenue > 0 else 0
    
    print(f"\n• Portfolio-wide impact (10% improvement):")
    print(f"  → Total sales increase: {portfolio_sales_impact_pct:.2f}% of all products")
    print(f"  → Total revenue increase: {portfolio_revenue_impact_pct:.2f}% of all revenue")
    
    return results


def _interpret_correlation(corr_coefficient: float, p_value: float) -> str:
    """
    Interpret correlation coefficient and p-value.
    
    Parameters:
    -----------
    corr_coefficient : float
        Correlation coefficient
    p_value : float
        P-value for statistical significance
        
    Returns:
    --------
    str
        Interpretation of the correlation
    """
    # Check statistical significance
    if p_value > 0.05:
        return "Not statistically significant"
    
    abs_corr = abs(corr_coefficient)
    
    if abs_corr < 0.1:
        return "Very weak correlation"
    elif abs_corr < 0.3:
        return "Weak correlation"
    elif abs_corr < 0.5:
        return "Moderate correlation"
    elif abs_corr < 0.7:
        return "Strong correlation"
    else:
        return "Very strong correlation"


def print_detailed_analysis(results: Dict[str, Any]) -> None:
    """
    Print detailed analysis from the results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from analyze_quality_improvement_impact function
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS REPORT")
    print("=" * 80)
    
    # Correlation analysis
    corr = results['correlation_analysis']['pearson']
    print(f"\n1. CORRELATION ANALYSIS")
    print(f"   Correlation coefficient: {corr['coefficient']:.4f}")
    print(f"   P-value: {corr['p_value']:.6f}")
    print(f"   Interpretation: {corr['interpretation']}")
    print(f"   Statistically significant: {'Yes' if corr['is_significant'] else 'No'}")
    print(f"   Samples analyzed: {results['correlation_analysis']['n_samples']:,}")
    
    # Summary statistics
    stats = results['summary_statistics']
    print(f"\n2. DATA SUMMARY")
    print(f"   Total products: {stats['total_products']:,}")
    print(f"   Low-quality products: {stats['low_quality_products']:,} ({stats['low_quality_percentage']:.1f}%)")
    print(f"   High-quality products: {stats['high_quality_products']:,}")
    print(f"   Average quality score (low): {stats['avg_quality_score_low']:.2f}")
    print(f"   Average quality score (high): {stats['avg_quality_score_high']:.2f}")
    print(f"   Average sales per product (low): {stats['avg_sales_low']:.2f}")
    print(f"   Average sales per product (high): {stats['avg_sales_high']:.2f}")
    
    # Low-quality products details
    low_quality = results['low_quality_data']
    print(f"\n3. LOW-QUALITY PRODUCTS DETAILS")
    print(f"   Number of products: {low_quality['product_count']:,}")
    print(f"   Current total sales: {low_quality['current_sales']:,.0f} units")
    print(f"   Current total revenue: ${low_quality['current_revenue']:,.2f}")
    
    # Get improvement table
    table = results['improvement_table_formatted']
    print(f"\n4. IMPROVEMENT TABLE (Formatted)")
    print("-" * 80)
    print(table.to_string(index=False))
    
    print("\n" + "=" * 80)

    
    return results

