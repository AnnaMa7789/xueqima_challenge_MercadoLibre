1. **Problem Understanding: Brief summary of the business problem**
In the context of e-commerce platform, product listing quality significantly impacts conversion rates and seller success. My task is to build a listing quality evaluation framework, based on listing and sales information for 100,000 products and to generate recommendation for those products identified as low listing quality.
2. **Methodology**:
Among 48 columns in the dataset, there are two main quantative metrics: price and sales (sold_quantity). As the product category varies from car to office supplies, I did not compare abosulte value, but instead using the price/sales ranking within each category to quantify each product's sales performance. Only category with more than 10 products are considered (~74%). Sales is the major output for quality scoring, but price is also relied upon to determine priority of fixing (for instance, if adding video link can boost 10% sales, it is recommended to work on high-value products first).

The remaining columns are qualitative metrics which provide detailed information of the product. Among them I picked the following key factors to build scoring systems. I have analyzed a few other metrics such as shipping info, picture quality, which did not make to the list due to minimal differentiation
**Title Length (characters)
Title Quality Score (depending on info density, key elements, placeholder words, repeated words,etc)
Whether it has video
Whether it has been updated
Number of Pictures
Attributes Entry Count
Attributes Completeness**

The weight of each factor is determined by correlation coefficient, in other words, how significant the impact of each factor on sales ranking within category.

With scoring system implemented, low (below 25th percentile) and median (25th ~ 50th percentile)  score products are identified. To achieve highest ROI, fix plan is provided for products with high sales increase potentail and with easy fix (title related issues)

4. **Key Findings**:
- Main insights from EDA
   *quantative metrics:
   The total sales are 239,699, which indicates a relatively low (2.39) sales per product.  Median price is 250, variation range is very large due to different product types. 99.99% of products never change price since listing is created.
   *qualitative metrics:
      - titles: there are 98,823 unique titles grouped under 10,907 categories. Average title length is 45 characters (ranging from 1 to 100). length of title has positive impact on sales with correlation coefficient = 0.625
video: only 2985 (~3%) products with video links. 
      - image: about 33.3% products have 1 picture, about 64.3% have 2-6 pictures. Only 789 products (~ 0.8%) are missing pictures. Majority of picture size is 500*375 or 500*500 and majority of max pricture size 1200*900
      - update frequency: ~70% of listings have never been updated since creation, ~30% are updated within 2 months.
      - attribute completeness: ~87% products have blank attributes, and ~10% have 1 or 2 entries. For those products who have at least one entries, about 80% have complete field information.  entry number (correlation  0.039) and completeness (correlation 0.091 ) both have positive impact on sales. 
      - shipping info completeness：all products have shipping and there is mininal differentiation regarding completeness. 
- Distribution of quality scores
Average quality score is 29.35 with standard deviation of 9.54. Score distribution as shown below indicates relative poor overall quality. Key drivers are attribute completeness and picture count, which show massive differences between low and high-quality listings. Quality scores vary significantly across categories, with some categories achieving much higher average quality. 
  Excellent (81-100)  :      0 products (  0.0%)
  Good (61-80)        :    733 products (  0.7%)
  Fair (41-60)        :   9807 products (  9.8%)
  Poor (21-40)        :  83474 products ( 83.5%)
  Very Poor (0-20)    :    747 products (  0.7%)

- Patterns identified
Sales are highly concentrated, with top performers driven by category selection and listing quality rather than price.Key drivers include high attribute completeness, multiple pictures, and updated listings, which strongly correlate with quality scores. However, many listings underperform due to poor quality and incomplete information, despite some having good individual metrics.
5. **Recommendations**:
Top 5 recommendations are: 
1. Implement mandatory attribute completion for key product fields
   Impact: Significant improvement in quality scores, better search relevance, and higher buyer confidence.
   Priority: HIGH
   Difficulty: MEDIUM
2. Set minimum picture requirements (3-5 pictures) for all listings
   Impact: Improved buyer experience, reduced returns, and higher quality scores.
   Priority: HIGH
   Difficulty: EASY
3. Create category-specific quality templates based on top-performing categories
   Impact: Category-wide quality improvements and more consistent standards.
   Priority: MEDIUM
   Difficulty: MEDIUM
4. Develop a 'Quality Improvement Wizard' that guides sellers through the most impactful improvements
   Impact: Rapid quality improvements with minimal seller effort.
   Priority: MEDIUM
   Difficulty: HARD
5. Establish quality score thresholds for search ranking and visibility
   Impact: Increased motivation for sellers to improve listings, leading to platform-wide quality improvements.
   Priority: MEDIUM
   Difficulty: HARD


based on the assumption that 
6. **GenAI Usage**:
Tools: deepseek
Here is the function to Use AI to analyze listings with low quality scores and provide recommendation

sample prompt:
   system_prompt = '''
You are an experienced e-commerce optimization specialist for MercadoLibre.
Your expertise is in identifying and fixing non-title related listing quality issues.

NON-TITLE QUALITY FACTORS INCLUDE:
1. Image quality and quantity (picture_count)
2. Video presence (has_video)
3. Listing freshness (has_updated)
4. Attributes completeness (attributes_count, attributes_completeness)
5. Other listing elements that affect conversion rates

Your task is to:
1. Analyze listings with good titles but low overall quality scores
2. Identify the specific non-title issues causing low scores
3. Provide actionable, specific recommendations
4. Focus on quick wins that sellers can implement immediately

'''
   input_prompt:input_prompt = f'''
LISTINGS DATA:
Total listings analyzed: {len(listings_data)}
All listings have: Good title scores (≥70) but Low overall quality scores (≤60)

SAMPLE LISTINGS DATA:
{json.dumps(listings_data[:20], ensure_ascii=False, indent=2)}

TASK:
Based on the non-title metrics provided, identify the 10 LISTINGS WITH THE MOST ACTIONABLE NON-TITLE ISSUES.
For each listing, provide:

1. SPECIFIC ISSUES IDENTIFIED: What non-title factors are causing low scores?
2. BUSINESS IMPACT: How do these issues affect sales/conversions?
3. ACTIONABLE RECOMMENDATIONS: Specific, practical steps to fix each issue
4. ESTIMATED IMPROVEMENT: How much quality score improvement is possible?
5. PRIORITY LEVEL: Based on impact and ease of implementation

FORMAT RESPONSE AS JSON:
{{
  "analysis_summary": {{
    "total_listings_analyzed": {len(listings_data)},
    "most_common_issues": ["list of top 3 most common non-title issues"],
    "quick_wins_available": "percentage of listings with easy fixes",
    "estimated_avg_improvement": "average quality score improvement possible"
  }},
  "top_10_problematic_listings": [
    {{
      "listing_id": "ID",
      "current_overall_score": "number",
      "current_title_score": "number",
      "main_non_title_issues": ["specific issues identified"],
      "business_impact_explanation": "how this affects sales",
      "specific_recommendations": [
        {{
          "action": "specific action to take",
          "reason": "why this helps",
          "difficulty": "easy/medium/hard",
          "expected_improvement": "points improvement"
        }}
      ],
      "overall_improvement_potential": "total points possible",
      "priority_level": "high/medium/low",
      "time_to_implement": "estimated time needed"
    }}
  ],
  "general_recommendations": {{
    "for_listings_with_few_images": ["recommendations"],
    "for_listings_without_videos": ["recommendations"],
    "for_listings_with_poor_attributes": ["recommendations"],
    "for_stale_listings": ["recommendations"]
  }}
}}

Focus on PRACTICAL, ACTIONABLE advice that sellers can implement without technical expertise.
'''
sample output: 
🚨 TOP 10 LISTINGS WITH NON-TITLE ISSUES:
========================================================================================================================

1. Listing ID: MLA578261979
   Current Score: 40.6
   Main Issues: Only 1 image, No video
                (+ 2 more)
   Top Recommendation: Add at least 3 more high-quality photos showing different angles, details, and product in use
   Improvement Potential: 15-23 points
   Priority: HIGH
   Time to Implement: 15-20 minutes

2. Listing ID: MLA583311895
   Current Score: 40.16
   Main Issues: Only 1 image, No video
                (+ 1 more)
   Top Recommendation: Add 4-5 photos showing front, back, close-up of fabric, size label, and on a model if possible
   Improvement Potential: 15-23 points
   Priority: HIGH
   Time to Implement: 15 minutes

3. Listing ID: MLA583879540
   Current Score: 40.18
   Main Issues: Only 1 image, No video
                (+ 1 more)
   Top Recommendation: Upload 4-5 images showing design details, fabric texture, tag with size, and full garment
   Improvement Potential: 15-21 points
   Priority: HIGH
   Time to Implement: 15 minutes

4. Listing ID: MLA582209130
   Current Score: 41.15
   Main Issues: Only 1 image, No video
                (+ 2 more)
   Top Recommendation: Add 3-4 additional photos showing embroidery/details, size tag, and different angles
   Improvement Potential: 15-23 points
   Priority: HIGH
   Time to Implement: 15 minutes

5. Listing ID: MLA583968337
   Current Score: 38.39
   Main Issues: Only 1 image, No video
                (+ 1 more)
   Top Recommendation: Add 4-5 photos showing tread pattern, sidewalls, DOT date code, and overall condition
   Improvement Potential: 15-23 points
   Priority: HIGH
   Time to Implement: 20 minutes

6. Listing ID: MLA576583331
   Current Score: 40.18
   Main Issues: Only 2 images, No video
                (+ 1 more)
   Top Recommendation: Add 3-4 more photos showing fabric close-up, pattern details, and on-model if possible
   Improvement Potential: 14-21 points
   Priority: HIGH
   Time to Implement: 15 minutes

7. Listing ID: MLA584564101
   Current Score: 41.69
   Main Issues: Only 3 images, No video
                (+ 2 more)
   Top Recommendation: Add 2-3 more photos showing part numbers, connection points, and installation orientation
   Improvement Potential: 16-25 points
   Priority: HIGH
   Time to Implement: 20 minutes

8. Listing ID: MLA584708202
   Current Score: 43.77
   Main Issues: No video, Only 1 attribute
                (+ 1 more)
   Top Recommendation: Add 10+ relevant attributes including part number, vehicle compatibility, dimensions, and condition details
   Improvement Potential: 17-26 points
   Priority: MEDIUM
   Time to Implement: 30 minutes

9. Listing ID: MLA576525056
   Current Score: 46.16
   Main Issues: Only 4 images, No video
                (+ 1 more)
   Top Recommendation: Add 2-3 more photos showing back detail, lining, and close-up of embellishments
   Improvement Potential: 10-16 points
   Priority: MEDIUM
   Time to Implement: 15 minutes

10. Listing ID: MLA574765459
   Current Score: 46.11
   Main Issues: Only 3 images, No video
                (+ 2 more)
   Top Recommendation: Add 2-3 more photos showing logo details, fabric texture, and care labels
   Improvement Potential: 12-20 points
   Priority: MEDIUM
   Time to Implement: 15 minutes

- Value delivered
This diagnostic assessment details the reasons behind low listing quality and outlines corrective action plans. Given the multitude of potential causes, AI efficiently filters and prioritizes the issues offering the highest incremental benefit—that is, those whose resolution would have a quantifiable impact on sales. Consequently, it not only generates considerable savings in time and labor but also ensures that efforts are channeled toward the most critical areas.

