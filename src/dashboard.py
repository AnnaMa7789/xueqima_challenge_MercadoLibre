import gradio as gr
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Plotly template
pio.templates.default = "plotly_white"

# Global variables - scored_df should already be loaded from previous steps
# scored_df = None  # This should be set from your previous analysis
current_filtered_df = None

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def initialize_dashboard():
    """
    Initialize dashboard with existing scored_df
    """
    global scored_df, current_filtered_df
    
    if scored_df is None or len(scored_df) == 0:
        return "❌ No data available. scored_df is empty or not loaded.", None, "No data available"
    
    # Use the existing scored_df
    current_filtered_df = scored_df.copy()
    
    # Generate overview message
    total_products = len(scored_df)
    categories = scored_df['category_id'].nunique()
    avg_quality = scored_df['quality_score'].mean()
    avg_sales = scored_df['sold_quantity'].mean()
    avg_price = scored_df['price'].mean()
    
    info_text = f"""
    ✅ Dashboard Initialized!
    
    📊 Dataset Overview:
    • Total Products: {total_products:,}
    • Unique Categories: {categories}
    • Average Quality Score: {avg_quality:.2f}/100
    • Average Sales: {avg_sales:.1f} units
    • Average Price: ${avg_price:.2f}
    
    📈 Quality Score Distribution:
    • Excellent (81-100): {((scored_df['quality_score'] >= 81) & (scored_df['quality_score'] <= 100)).sum():,}
    • Good (61-80): {((scored_df['quality_score'] >= 61) & (scored_df['quality_score'] < 81)).sum():,}
    • Fair (41-60): {((scored_df['quality_score'] >= 41) & (scored_df['quality_score'] < 61)).sum():,}
    • Poor (21-40): {((scored_df['quality_score'] >= 21) & (scored_df['quality_score'] < 41)).sum():,}
    • Very Poor (0-20): {(scored_df['quality_score'] < 21).sum():,}
    """
    
    # Create overview plots
    overview_plot = create_overview_plots(scored_df)
    
    # Generate AI recommendations
    ai_recommendations_text = generate_initial_ai_recommendations(scored_df)
    
    return info_text, overview_plot, ai_recommendations_text

def create_overview_plots(df):
    """
    Create overview plots for the dashboard
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to visualize
        
    Returns:
    --------
    plotly.Figure : Combined plotly figure
    """
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Quality Score Distribution', 'Sales vs Quality',
                       'Price Distribution by Quality', 'Top Categories by Avg Quality'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Quality Score Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df['quality_score'],
            nbinsx=20,
            marker_color='steelblue',
            opacity=0.7,
            name='Quality Scores'
        ),
        row=1, col=1
    )
    
    # 2. Sales vs Quality (Scatter plot)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    fig.add_trace(
        go.Scatter(
            x=sample_df['quality_score'],
            y=sample_df['sold_quantity'],
            mode='markers',
            marker=dict(
                size=8,
                color=sample_df['price'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Price")
            ),
            text=sample_df['title'],
            hovertemplate='<b>%{text}</b><br>Quality: %{x:.1f}<br>Sales: %{y}<br>Price: $%{marker.color:.2f}<extra></extra>',
            name='Sales vs Quality'
        ),
        row=1, col=2
    )
    
    # 3. Price Distribution by Quality Tier (Box plot)
    df['quality_tier'] = pd.cut(df['quality_score'], 
                                bins=[0, 40, 60, 80, 100],
                                labels=['Poor (0-40)', 'Fair (41-60)', 'Good (61-80)', 'Excellent (81-100)'])
    
    for tier in df['quality_tier'].cat.categories:
        tier_data = df[df['quality_tier'] == tier]
        if len(tier_data) > 0:
            fig.add_trace(
                go.Box(
                    y=tier_data['price'],
                    name=str(tier),
                    boxpoints='outliers',
                    marker_color='lightseagreen'
                ),
                row=2, col=1
            )
    
    # 4. Top Categories by Average Quality (Bar chart)
    if 'category_id' in df.columns and len(df['category_id'].unique()) > 0:
        top_categories = df.groupby('category_id')['quality_score'].agg(['mean', 'count']).round(2)
        top_categories = top_categories.sort_values('mean', ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=top_categories.index,
                y=top_categories['mean'],
                text=top_categories['mean'].round(1),
                textposition='auto',
                marker_color='coral',
                name='Avg Quality',
                hovertemplate='Category: %{x}<br>Avg Quality: %{y:.1f}<br>Count: %{customdata}<extra></extra>',
                customdata=top_categories['count']
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Product Listing Quality Dashboard Overview",
        title_font_size=16
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Quality Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_xaxes(title_text="Quality Score", row=1, col=2)
    fig.update_yaxes(title_text="Sold Quantity", row=1, col=2)
    
    fig.update_xaxes(title_text="Quality Tier", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    
    fig.update_xaxes(title_text="Category", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Average Quality Score", row=2, col=2)
    
    return fig

def generate_initial_ai_recommendations(df):
    """
    Generate initial AI recommendations based on data analysis
    """
    if df is None or len(df) == 0:
        return "No data available for AI analysis"
    
    # Calculate key metrics
    total_products = len(df)
    avg_quality = df['quality_score'].mean()
    
    # Identify common issues
    issues = []
    recommendations = []
    
    # Check for title issues
    if 'title_length' in df.columns:
        short_titles = (df['title_length'] < 30).sum()
        if short_titles > total_products * 0.1:
            issues.append(f"{short_titles} products have very short titles (<30 characters)")
            recommendations.append("• Add more descriptive keywords to short titles")
    
    # Check for picture issues
    if 'picture_count' in df.columns:
        few_pics = (df['picture_count'] < 3).sum()
        if few_pics > 0:
            issues.append(f"{few_pics} products have less than 3 pictures")
            recommendations.append("• Increase image count to at least 3-5 per product")
    
    # Check for video presence
    if 'has_video' in df.columns:
        no_video = (df['has_video'] == 0).sum()
        if no_video > total_products * 0.5:
            issues.append(f"{no_video} products have no video content")
            recommendations.append("• Add product demonstration videos for top-selling items")
    
    # Check for attribute completeness
    if 'attr_completeness_pct' in df.columns:
        incomplete_attrs = (df['attr_completeness_pct'] < 80).sum()
        if incomplete_attrs > 0:
            issues.append(f"{incomplete_attrs} products have incomplete attributes (<80%)")
            recommendations.append("• Complete attribute fields for all products")
    
    # Build recommendations text
    recommendations_text = f"""
    🤖 AI-PRODUCED RECOMMENDATIONS
    
    📊 Data Summary:
    • Total Products Analyzed: {total_products:,}
    • Average Quality Score: {avg_quality:.2f}/100
    
    🔍 Identified Issues:
    {chr(10).join([f'• {issue}' for issue in issues]) if issues else '• No major issues identified'}
    
    💡 Top Recommendations:
    {chr(10).join(recommendations) if recommendations else '• All products meet basic quality standards'}
    
    🎯 Priority Actions:
    1. Focus on products with quality scores below 50
    2. Optimize titles with low title scores
    3. Improve images for products with fewer than 3 pictures
    4. Add videos to high-value products
    5. Complete all attribute fields
    """
    
    return recommendations_text

def apply_filters(category_filter, quality_min, quality_max, 
                  sales_min, sales_max, price_min, price_max,
                  has_video_filter, has_updated_filter):
    """
    Apply filters to the dataset
    """
    global scored_df, current_filtered_df
    
    if scored_df is None:
        return "No data loaded", None, None
    
    # Start with all data
    filtered_df = scored_df.copy()
    
    # Apply category filter
    if category_filter and category_filter != "All":
        filtered_df = filtered_df[filtered_df['category_id'] == category_filter]
    
    # Apply quality score range
    filtered_df = filtered_df[
        (filtered_df['quality_score'] >= quality_min) & 
        (filtered_df['quality_score'] <= quality_max)
    ]
    
    # Apply sales range
    filtered_df = filtered_df[
        (filtered_df['sold_quantity'] >= sales_min) & 
        (filtered_df['sold_quantity'] <= sales_max)
    ]
    
    # Apply price range
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_min) & 
        (filtered_df['price'] <= price_max)
    ]
    
    # Apply video filter
    if has_video_filter != "All":
        has_video_bool = has_video_filter == "Yes"
        if 'has_video' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['has_video'] == has_video_bool]
    
    # Apply update filter
    if has_updated_filter != "All":
        has_updated_bool = has_updated_filter == "Yes"
        if 'has_updated' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['has_updated'] == has_updated_bool]
    
    # Store filtered dataframe
    current_filtered_df = filtered_df.copy()
    
    # Generate filter info
    filter_info = f"""
    🔍 Applied Filters:
    • Products remaining: {len(filtered_df):,} of {len(scored_df):,}
    • Quality range: {quality_min}-{quality_max}
    • Sales range: {sales_min}-{sales_max}
    • Price range: ${price_min}-${price_max}
    • Category: {category_filter if category_filter != 'All' else 'All'}
    • Has Video: {has_video_filter}
    • Has Updated: {has_updated_filter}
    """
    
    # Create filtered plot
    filtered_plot = create_filtered_visualization(filtered_df)
    
    # Create listings table (show top 100 for performance)
    display_df = filtered_df.head(100).copy()
    
    # Select columns for display
    display_columns = ['id', 'title', 'category_id', 'quality_score', 
                      'price', 'sold_quantity', 'picture_count']
    
    # Add available columns
    available_cols = [col for col in display_columns if col in display_df.columns]
    
    listings_table = display_df[available_cols]
    
    return filter_info, filtered_plot, listings_table

def create_filtered_visualization(df):
    """
    Create visualization for filtered data
    """
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data after filtering",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create quality distribution for filtered data
    fig = go.Figure()
    
    # Add histogram for quality scores
    fig.add_trace(go.Histogram(
        x=df['quality_score'],
        nbinsx=20,
        marker_color='lightcoral',
        opacity=0.7,
        name='Filtered Quality Scores'
    ))
    
    # Add vertical line for average
    avg_quality = df['quality_score'].mean()
    fig.add_vline(x=avg_quality, line_dash="dash", line_color="red", 
                  annotation_text=f"Avg: {avg_quality:.1f}")
    
    # Update layout
    fig.update_layout(
        title=f"Quality Distribution (Filtered: {len(df)} products)",
        xaxis_title="Quality Score",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    return fig

def get_listing_details(product_id):
    """
    Get detailed information for a specific product
    """
    global scored_df
    
    if scored_df is None:
        return "No data loaded", "No data loaded", "No data loaded"
    
    # Find the product
    if 'id' in scored_df.columns:
        product_row = scored_df[scored_df['id'].astype(str) == str(product_id)]
    else:
        return "ID column not found", "", ""
    
    if len(product_row) == 0:
        return f"Product {product_id} not found", "", ""
    
    product = product_row.iloc[0]
    
    # Create basic info
    basic_info = f"""
    📋 PRODUCT DETAILS - ID: {product_id}
    {'='*50}
    
    📝 Basic Information:
    • Title: {product.get('title', 'N/A')[:100]}
    • Category: {product.get('category_id', 'N/A')}
    • Price: ${product.get('price', 0):.2f}
    • Sales: {product.get('sold_quantity', 0):,}
    • Available Quantity: {product.get('available_quantity', 'N/A')}
    
    🖼️ Media & Content:
    • Pictures: {product.get('picture_count', 'N/A')}
    • Has Video: {'Yes' if product.get('has_video', False) else 'No'}
    • Title Length: {product.get('title_length', 'N/A')} characters
    • Title Score: {product.get('title_score', 'N/A')}
    
    📊 Attributes:
    • Attribute Entries: {product.get('attr_entries', 'N/A')}
    • Attribute Completeness: {product.get('attr_completeness_pct', 'N/A')}%
    
    ⏰ Listing Status:
    • Created: {product.get('date_created', 'N/A')}
    • Last Updated: {product.get('last_updated', 'N/A')}
    • Has Been Updated: {'Yes' if product.get('has_updated', False) else 'No'}
    """
    
    # Create quality breakdown
    quality_score = product.get('quality_score', 0)
    quality_breakdown = f"""
    🎯 QUALITY ANALYSIS
    {'='*50}
    
    Overall Quality Score: {quality_score}/100
    
    Score Breakdown:
    """
    
    if 'score_breakdown' in product and isinstance(product['score_breakdown'], dict):
        breakdown = product['score_breakdown']
        for key, data in breakdown.items():
            if isinstance(data, dict):
                raw_score = data.get('raw_score', 0)
                weight = data.get('weight', 0)
                weighted_score = data.get('weighted_score', 0)
                
                quality_breakdown += f"\n• {key}:"
                quality_breakdown += f"\n  Raw Score: {raw_score:.1f}"
                quality_breakdown += f"\n  Weight: {weight:.1f}%"
                quality_breakdown += f"\n  Contribution: {weighted_score:.2f}"
    else:
        # Use individual scores if available
        score_sections = []
        if 'title_score' in product and pd.notna(product['title_score']):
            score_sections.append(f"• Title Quality: {product['title_score']:.1f}")
        
        if 'picture_count' in product and pd.notna(product['picture_count']):
            pic_score = min(product['picture_count'] * 10, 100)
            score_sections.append(f"• Images: {pic_score:.1f} ({product['picture_count']} pics)")
        
        if 'has_video' in product and pd.notna(product['has_video']):
            video_score = 100 if product['has_video'] else 0
            score_sections.append(f"• Video: {video_score:.1f}")
        
        if 'attr_completeness_pct' in product and pd.notna(product['attr_completeness_pct']):
            score_sections.append(f"• Attributes: {product['attr_completeness_pct']:.1f}")
        
        quality_breakdown += "\n" + "\n".join(score_sections)
    
    # Generate recommendation based on scores
    recommendation = generate_product_recommendation(product)
    
    return basic_info, quality_breakdown, recommendation

def generate_product_recommendation(product):
    """
    Generate specific recommendations for a product
    """
    recommendations = []
    
    # Check title
    title_score = product.get('title_score', 0)
    if title_score < 70:
        recommendations.append("📝 **Improve Title:** Add more descriptive keywords and product specifications")
    
    # Check pictures
    picture_count = product.get('picture_count', 0)
    if picture_count < 3:
        recommendations.append(f"🖼️ **Add More Images:** Current {picture_count} images, recommend at least 3-5")
    
    # Check video
    if not product.get('has_video', False):
        recommendations.append("🎬 **Add Video:** Product demonstration video can increase conversion by 30%")
    
    # Check attributes
    attr_completeness = product.get('attr_completeness_pct', 100)
    if attr_completeness < 80:
        recommendations.append(f"📋 **Complete Attributes:** Currently {attr_completeness:.1f}% complete, aim for 100%")
    
    # Check update status
    if not product.get('has_updated', True):
        recommendations.append("🔄 **Update Listing:** Refresh the listing with current information")
    
    # Check quality score
    quality_score = product.get('quality_score', 0)
    if quality_score < 50:
        recommendations.append(f"⚠️ **Priority Improvement:** Quality score {quality_score:.1f} needs immediate attention")
    elif quality_score < 70:
        recommendations.append(f"📈 **Moderate Improvement:** Quality score {quality_score:.1f} has room for improvement")
    else:
        recommendations.append(f"✅ **Good Quality:** Maintain current standards at {quality_score:.1f} score")
    
    # Format recommendations
    if recommendations:
        recommendations_text = "💡 RECOMMENDATIONS FOR IMPROVEMENT\n" + "="*50 + "\n\n"
        for i, rec in enumerate(recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"
    else:
        recommendations_text = "✅ This product meets all quality standards. No specific recommendations needed."
    
    return recommendations_text

# ============================================================================
# AI ANALYSIS FUNCTIONS
# ============================================================================

def analyze_quality_patterns(df):
    """Analyze quality score patterns"""
    
    total_products = len(df)
    avg_quality = df['quality_score'].mean()
    
    # Calculate distribution
    quality_bins = pd.cut(df['quality_score'], 
                         bins=[0, 20, 40, 60, 80, 100],
                         labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
    distribution = quality_bins.value_counts()
    
    # Find correlations
    correlations = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'quality_score':
            try:
                corr = df['quality_score'].corr(df[col])
                if not pd.isna(corr):
                    correlations[col] = corr
            except:
                continue
    
    # Sort by absolute correlation
    top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    analysis = f"""
    🤖 AI ANALYSIS: QUALITY PATTERNS
    {'='*60}
    
    📊 Overview:
    • Total Products: {total_products:,}
    • Average Quality Score: {avg_quality:.2f}/100
    
    📈 Quality Distribution:
    """
    
    for category, count in distribution.items():
        percentage = (count / total_products) * 100
        analysis += f"• {category}: {count:,} products ({percentage:.1f}%)\n"
    
    analysis += f"\n🔗 Top Correlations with Quality Score:\n"
    for col, corr in top_correlations:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        analysis += f"• {col}: {corr:.3f} ({strength} {direction} correlation)\n"
    
    # Key insights
    analysis += f"\n💡 KEY INSIGHTS:\n"
    
    if avg_quality < 50:
        analysis += "1. ⚠️ Overall quality is low - immediate improvement needed\n"
    elif avg_quality < 70:
        analysis += "1. 📈 Moderate quality - room for improvement\n"
    else:
        analysis += "1. ✅ Good overall quality - focus on maintaining standards\n"
    
    # Find most common issues
    issues = []
    if 'picture_count' in df.columns and df['picture_count'].mean() < 3:
        issues.append("Low image count")
    if 'has_video' in df.columns and df['has_video'].mean() < 0.2:
        issues.append("Lack of video content")
    if 'attr_completeness_pct' in df.columns and df['attr_completeness_pct'].mean() < 80:
        issues.append("Incomplete attributes")
    
    if issues:
        analysis += f"2. 🔍 Common issues: {', '.join(issues)}\n"
    
    return analysis

def analyze_sales_drivers(df):
    """Analyze factors driving sales"""
    
    if 'sold_quantity' not in df.columns:
        return "Sales data not available for analysis"
    
    total_sales = df['sold_quantity'].sum()
    avg_sales = df['sold_quantity'].mean()
    
    # Calculate correlations with sales
    correlations = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'sold_quantity':
            try:
                corr = df['sold_quantity'].corr(df[col])
                if not pd.isna(corr):
                    correlations[col] = corr
            except:
                continue
    
    # Sort by absolute correlation
    top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Analyze by quality tier
    df['quality_tier'] = pd.cut(df['quality_score'], 
                               bins=[0, 40, 60, 80, 100],
                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    tier_sales = df.groupby('quality_tier')['sold_quantity'].agg(['mean', 'sum', 'count']).round(2)
    
    analysis = f"""
    🤖 AI ANALYSIS: SALES DRIVERS
    {'='*60}
    
    📊 Sales Overview:
    • Total Sales: {total_sales:,} units
    • Average Sales per Product: {avg_sales:.1f} units
    • Total Products: {len(df):,}
    
    🔗 Top Correlations with Sales:
    """
    
    for col, corr in top_correlations:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        analysis += f"• {col}: {corr:.3f} ({strength} {direction} correlation)\n"
    
    analysis += f"\n📈 Sales by Quality Tier:\n"
    for tier, row in tier_sales.iterrows():
        analysis += f"• {tier}: Avg {row['mean']:.1f} sales, Total {row['sum']:,} units ({row['count']} products)\n"
    
    # Price-sales relationship
    if 'price' in df.columns:
        price_sales_corr = df['sold_quantity'].corr(df['price'])
        analysis += f"\n💰 Price-Sales Relationship:\n"
        analysis += f"• Correlation: {price_sales_corr:.3f}\n"
        
        if price_sales_corr < -0.2:
            analysis += "• Insight: Strong negative correlation - lower prices drive higher sales\n"
        elif price_sales_corr > 0.2:
            analysis += "• Insight: Strong positive correlation - higher prices associated with higher sales\n"
        else:
            analysis += "• Insight: Weak correlation - price not a major sales driver\n"
    
    return analysis

def analyze_improvement_opportunities(df):
    """Analyze improvement opportunities"""
    
    total_products = len(df)
    
    # Identify low-hanging fruit
    opportunities = []
    
    # 1. Low quality scores
    low_quality = df[df['quality_score'] < 50]
    if len(low_quality) > 0:
        opportunities.append(f"{len(low_quality)} products with quality scores < 50")
    
    # 2. Few images
    if 'picture_count' in df.columns:
        few_images = df[df['picture_count'] < 3]
        if len(few_images) > 0:
            opportunities.append(f"{len(few_images)} products with less than 3 images")
    
    # 3. No video
    if 'has_video' in df.columns:
        no_video = df[df['has_video'] == 0] 
        if len(no_video) > 0:
            opportunities.append(f"{len(no_video)} products without video")
    
    # 4. Incomplete attributes
    if 'attr_completeness_pct' in df.columns:
        incomplete = df[df['attr_completeness_pct'] < 80]
        if len(incomplete) > 0:
            opportunities.append(f"{len(incomplete)} products with incomplete attributes (<80%)")
    
    # 5. Never updated
    if 'has_updated' in df.columns:
        never_updated = df[df['has_updated'] == 0] 
        if len(never_updated) > 0:
            opportunities.append(f"{len(never_updated)} listings never updated")
    
    analysis = f"""
    🤖 AI ANALYSIS: IMPROVEMENT OPPORTUNITIES
    {'='*60}
    
    📊 Overview:
    • Total Products Analyzed: {total_products:,}
    
    🎯 Improvement Opportunities Found:
    """
    
    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            analysis += f"{i}. {opp}\n"
    else:
        analysis += "✅ No major improvement opportunities identified\n"
    
    # Priority recommendations
    analysis += f"\n💡 PRIORITY RECOMMENDATIONS:\n"
    
    if 'low_quality' in locals() and len(low_quality) > 0:
        analysis += f"1. 🔴 HIGH PRIORITY: Focus on {len(low_quality)} products with quality scores < 50\n"
        analysis += f"   • These represent {(len(low_quality)/total_products)*100:.1f}% of products\n"
        analysis += f"   • Expected improvement: 20-40 points per product\n"
    
    if 'few_images' in locals() and len(few_images) > 0:
        analysis += f"\n2. 🟡 MEDIUM PRIORITY: Add images to {len(few_images)} products\n"
        analysis += f"   • Quick win: Adding 2-3 images can improve scores by 10-20 points\n"
    
    if 'no_video' in locals() and len(no_video) > 0:
        analysis += f"\n3. 🟢 LOW PRIORITY: Consider adding videos\n"
        analysis += f"   • Start with top-selling products first\n"
        analysis += f"   • Videos can increase conversion by 30-40%\n"
    
    return analysis

def analyze_categories(df):
    """Analyze category performance"""
    
    if 'category_id' not in df.columns:
        return "Category data not available"
    
    # Calculate category statistics
    category_stats = df.groupby('category_id').agg({
        'id': 'count',
        'quality_score': ['mean', 'min', 'max'],
        'sold_quantity': ['sum', 'mean'],
        'price': 'mean'
    }).round(2)
    
    # Flatten column names
    category_stats.columns = ['product_count', 'avg_quality', 'min_quality', 'max_quality', 
                             'total_sales', 'avg_sales', 'avg_price']
    category_stats = category_stats.reset_index()
    
    total_categories = len(category_stats)
    
    # Find best and worst categories
    best_quality = category_stats.nlargest(3, 'avg_quality')
    worst_quality = category_stats.nsmallest(3, 'avg_quality')
    best_sales = category_stats.nlargest(3, 'total_sales')
    
    analysis = f"""
    🤖 AI ANALYSIS: CATEGORY PERFORMANCE
    {'='*60}
    
    📊 Overview:
    • Total Categories: {total_categories}
    • Total Products: {len(df):,}
    
    🏆 Top Categories by Quality:
    """
    
    for i, (_, row) in enumerate(best_quality.iterrows(), 1):
        analysis += f"{i}. {row['category_id']}: {row['avg_quality']:.1f} avg quality ({row['product_count']} products)\n"
    
    analysis += f"\n⚠️ Categories Needing Improvement:\n"
    for i, (_, row) in enumerate(worst_quality.iterrows(), 1):
        analysis += f"{i}. {row['category_id']}: {row['avg_quality']:.1f} avg quality ({row['product_count']} products)\n"
    
    analysis += f"\n💰 Top Categories by Sales:\n"
    for i, (_, row) in enumerate(best_sales.iterrows(), 1):
        analysis += f"{i}. {row['category_id']}: {row['total_sales']:,} total sales (${row['avg_price']:.2f} avg price)\n"
    
    # Category insights
    analysis += f"\n💡 CATEGORY INSIGHTS:\n"
    
    # Quality range across categories
    quality_range = category_stats['avg_quality'].max() - category_stats['avg_quality'].min()
    analysis += f"1. Quality varies significantly across categories (range: {quality_range:.1f} points)\n"
    
    # Identify opportunities
    if len(worst_quality) > 0:
        worst_cat = worst_quality.iloc[0]
        analysis += f"2. Priority: Focus on '{worst_cat['category_id']}' category\n"
        analysis += f"   • Current avg quality: {worst_cat['avg_quality']:.1f}\n"
        analysis += f"   • {worst_cat['product_count']} products in this category\n"
        analysis += f"   • Target: Improve to at least 60+ points\n"
    
    # Category with highest potential
    if len(best_sales) > 0 and len(best_quality) > 0:
        high_sales_cat = best_sales.iloc[0]
        high_quality_cat = best_quality.iloc[0]
        
        if high_sales_cat['category_id'] != high_quality_cat['category_id']:
            analysis += f"\n3. Opportunity: '{high_sales_cat['category_id']}' has high sales but moderate quality\n"
            analysis += f"   • Sales: {high_sales_cat['total_sales']:,} units\n"
            analysis += f"   • Quality: {high_sales_cat['avg_quality']:.1f} points\n"
            analysis += f"   • Improving quality here could significantly boost sales\n"
    
    return analysis

def generate_ai_analysis(analysis_type):
    """
    Generate AI analysis based on selected type
    """
    global current_filtered_df
    
    if current_filtered_df is None or len(current_filtered_df) == 0:
        return "No data available for AI analysis"
    
    df = current_filtered_df
    
    # Based on analysis type, generate different insights
    if analysis_type == "Quality Patterns":
        return analyze_quality_patterns(df)
    elif analysis_type == "Sales Drivers":
        return analyze_sales_drivers(df)
    elif analysis_type == "Improvement Opportunities":
        return analyze_improvement_opportunities(df)
    elif analysis_type == "Category Analysis":
        return analyze_categories(df)
    else:
        return "Select an analysis type"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_category_dropdown():
    """
    Update category dropdown choices based on scored_df
    """
    global scored_df
    if scored_df is not None and 'category_id' in scored_df.columns:
        categories = ["All"] + sorted(scored_df['category_id'].unique().tolist())
        return gr.update(choices=categories, value="All")
    return gr.update(choices=["All"], value="All")

def reset_filters():
    """
    Reset all filter values to default
    """
    return (
        0, 100,  # quality range
        0, 10000,  # sales range
        0, 10000,  # price range
        "All", "All", "All"  # dropdowns
    )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """
    Create the Gradio web interface using existing scored_df
    """
    
    # Define theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    )
    
    # Create tabs
    with gr.Blocks(theme=theme, title="Product Listing Quality Analyzer") as demo:
        gr.Markdown("# 📊 Product Listing Quality Analyzer")
        gr.Markdown("### Analyze and optimize your MercadoLibre product listings")
        
        with gr.Tabs():
            # Tab 1: Overview Dashboard
            with gr.Tab("📊 Overview Dashboard"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Dashboard Control")
                        initialize_btn = gr.Button("🚀 Initialize Dashboard", variant="primary")
                        refresh_btn = gr.Button("🔄 Refresh")
                        
                        data_info = gr.Textbox(
                            label="Dataset Information",
                            lines=15,
                            interactive=False
                        )
                        
                        gr.Markdown("### 📝 Instructions")
                        gr.Markdown("""
                        1. Click **Initialize Dashboard** to start
                        2. Use other tabs to explore and analyze data
                        3. Data is already loaded from scored_df
                        """)
                        
                    with gr.Column(scale=3):
                        overview_plot = gr.Plot(
                            label="Overview Dashboard",
                            show_label=True
                        )
                
                with gr.Row():
                    ai_recommendations = gr.Textbox(
                        label="🤖 AI Recommendations",
                        lines=10,
                        interactive=False
                    )
            
            # Tab 2: Filter & Explore
            with gr.Tab("🔍 Filter & Explore"):
                with gr.Row():
                    # Filters column
                    with gr.Column(scale=1):
                        gr.Markdown("### Filters")
                        
                        category_dropdown = gr.Dropdown(
                            label="Category",
                            choices=["All"],
                            value="All"
                        )
                        
                        quality_min = gr.Slider(
                            label="Min Quality Score",
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1
                        )
                        
                        quality_max = gr.Slider(
                            label="Max Quality Score",
                            minimum=0,
                            maximum=100,
                            value=100,
                            step=1
                        )
                        
                        sales_min = gr.Slider(
                            label="Min Sales Quantity",
                            minimum=0,
                            maximum=10000,
                            value=0,
                            step=10
                        )
                        
                        sales_max = gr.Slider(
                            label="Max Sales Quantity",
                            minimum=0,
                            maximum=10000,
                            value=10000,
                            step=10
                        )
                        
                        price_min = gr.Slider(
                            label="Min Price ($)",
                            minimum=0,
                            maximum=10000,
                            value=0,
                            step=10
                        )
                        
                        price_max = gr.Slider(
                            label="Max Price ($)",
                            minimum=0,
                            maximum=10000,
                            value=10000,
                            step=10
                        )
                        
                        has_video_filter = gr.Radio(
                            label="Has Video",
                            choices=["All", "Yes", "No"],
                            value="All"
                        )
                        
                        has_updated_filter = gr.Radio(
                            label="Has Updated",
                            choices=["All", "Yes", "No"],
                            value="All"
                        )
                        
                        apply_filters_btn = gr.Button(
                            "🔍 Apply Filters",
                            variant="primary"
                        )
                        
                        reset_filters_btn = gr.Button("🔄 Reset Filters")
                    
                    # Results column
                    with gr.Column(scale=2):
                        filter_info = gr.Textbox(
                            label="Filter Results",
                            lines=5,
                            interactive=False
                        )
                        
                        filtered_plot = gr.Plot(
                            label="Filtered Distribution"
                        )
                        
                        listings_table = gr.Dataframe(
                            label="Filtered Listings (Top 100)",
                            interactive=False,
                            elem_id="listings-table")

                        # 然后在 CSS 中设置高度
                        css = """
                        #listings-table {
                            height: 400px !important;
                        }
                        """            
            # Tab 3: Listing Details
            with gr.Tab("📋 Listing Details"):
                with gr.Row():
                    with gr.Column(scale=1):
                        product_id_input = gr.Textbox(
                            label="Enter Product ID",
                            placeholder="e.g., 12345"
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary")
                        
                        gr.Markdown("### Search Tips")
                        gr.Markdown("""
                        - Enter the exact product ID
                        - IDs are usually numeric
                        - Case-sensitive
                        """)
                    
                    with gr.Column(scale=2):
                        basic_info = gr.Textbox(
                            label="Basic Information",
                            lines=15,
                            interactive=False
                        )
                        
                        quality_breakdown = gr.Textbox(
                            label="Quality Breakdown",
                            lines=10,
                            interactive=False
                        )
                        
                        recommendation = gr.Textbox(
                            label="Recommendations",
                            lines=8,
                            interactive=False
                        )
            
            # Tab 4: AI Analysis
            with gr.Tab("🤖 AI Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analysis_type = gr.Dropdown(
                            label="Select Analysis Type",
                            choices=[
                                "Quality Patterns",
                                "Sales Drivers", 
                                "Improvement Opportunities",
                                "Category Analysis"
                            ],
                            value="Quality Patterns"
                        )
                        
                        generate_btn = gr.Button(
                            "🚀 Generate AI Analysis",
                            variant="primary"
                        )
                        
                        gr.Markdown("### Analysis Types")
                        gr.Markdown("""
                        - **Quality Patterns**: Analyze quality score distributions
                        - **Sales Drivers**: Identify factors affecting sales
                        - **Improvement Opportunities**: Find optimization areas
                        - **Category Analysis**: Compare performance across categories
                        """)
                    
                    with gr.Column(scale=2):
                        ai_analysis_output = gr.Textbox(
                            label="AI Analysis Results",
                            lines=30,
                            interactive=False
                        )
        
        # Event handlers
        # Initialize dashboard
        initialize_btn.click(
            fn=initialize_dashboard,
            inputs=[],
            outputs=[data_info, overview_plot, ai_recommendations]
        ).then(
            fn=update_category_dropdown,
            outputs=[category_dropdown]
        )
        
        # Refresh dashboard
        refresh_btn.click(
            fn=initialize_dashboard,
            inputs=[],
            outputs=[data_info, overview_plot, ai_recommendations]
        )
        
        # Apply filters
        apply_filters_btn.click(
            fn=apply_filters,
            inputs=[
                category_dropdown,
                quality_min, quality_max,
                sales_min, sales_max,
                price_min, price_max,
                has_video_filter,
                has_updated_filter
            ],
            outputs=[filter_info, filtered_plot, listings_table]
        )
        
        # Reset filters
        reset_filters_btn.click(
            fn=reset_filters,
            outputs=[
                quality_min, quality_max,
                sales_min, sales_max,
                price_min, price_max,
                category_dropdown,
                has_video_filter,
                has_updated_filter
            ]
        )
        
        # Search product details
        search_btn.click(
            fn=get_listing_details,
            inputs=[product_id_input],
            outputs=[basic_info, quality_breakdown, recommendation]
        )
        
        # Generate AI analysis
        generate_btn.click(
            fn=generate_ai_analysis,
            inputs=[analysis_type],
            outputs=[ai_analysis_output]
        )
    
    return demo

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_dashboard(existing_scored_df):
    """
    Run the Gradio dashboard with existing scored_df
    
    Parameters:
    -----------
    existing_scored_df : pandas DataFrame
        The scored_df from previous analysis
    """
    global scored_df
    
    # Set the global scored_df
    scored_df = existing_scored_df
    
    if scored_df is None or len(scored_df) == 0:
        print("❌ Error: scored_df is empty or not provided")
        return
    
    print(f"✅ Loaded scored_df with {len(scored_df):,} products")
    print(f"📊 Available columns: {len(scored_df.columns)}")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    print("\n🚀 Starting Gradio dashboard...")
    print("🎯 Click 'Initialize Dashboard' to begin")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=False
    )
