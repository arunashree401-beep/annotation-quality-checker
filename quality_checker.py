# ============================================================
# ANNOTATION QUALITY CHECKER
# Author: Arunashree S
# Purpose: Automate ML annotation quality checks
#          Inter-annotator agreement, error detection,
#          and quality report generation
# ============================================================

# --- IMPORTS ---
# pandas  : helps us read and work with CSV/table data
# sklearn : gives us Cohen's Kappa score calculation
# matplotlib / seaborn : helps us create charts and graphs
# os      : helps us work with file paths and folders
# datetime: helps us add timestamps to reports

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


# ============================================================
# STEP 1 — LOAD DATA
# This function reads the CSV file and loads it into a table
# called a "DataFrame" (df). Think of it like opening Excel.
# ============================================================

def load_data(filepath):
    """
    Loads annotation CSV file into a pandas DataFrame.
    
    filepath : the location of the CSV file on your computer
    returns  : a DataFrame (table) with all annotation data
    """
    print(f"\n📂 Loading data from: {filepath}")
    
    # pd.read_csv reads the CSV file into a table
    df = pd.read_csv(filepath)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Total records found: {len(df)}")        # how many rows
    print(f"   Total columns found: {len(df.columns)}") # how many columns
    print(f"   Column names: {list(df.columns)}\n")    # column names
    
    return df


# ============================================================
# STEP 2 — CHECK FOR MISSING VALUES
# Missing values = empty cells in the CSV
# In annotation work, empty = annotator didn't label something
# This is a quality problem we need to flag
# ============================================================

def check_missing_values(df):
    """
    Finds all empty/missing cells in the annotation data.
    
    df      : our DataFrame (table of annotations)
    returns : a summary dictionary of missing value counts
    """
    print("=" * 55)
    print("🔍 CHECKING FOR MISSING VALUES")
    print("=" * 55)
    
    # isnull() returns True/False for each cell (True = empty)
    # sum() counts how many True values exist per column
    missing = df.isnull().sum()
    
    # Calculate percentage of missing values per column
    missing_percent = (missing / len(df)) * 100
    
    # Create a summary table
    missing_summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent.round(2)
    })
    
    # Only show columns that actually have missing values
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
    
    if len(missing_summary) == 0:
        print("✅ No missing values found! Great data quality.\n")
    else:
        print(f"⚠️  Missing values found in {len(missing_summary)} column(s):\n")
        print(missing_summary.to_string())  # print the table nicely
        print()
    
    return missing_summary


# ============================================================
# STEP 3 — CHECK ANNOTATION CONSISTENCY
# Consistency = do all annotators agree on the same label?
# Example: If annotator_1 says "cat" and annotator_2 says "dog"
#          for the same image — that's an inconsistency!
# ============================================================

def check_consistency(df):
    """
    Checks if annotators agree with each other for each image.
    
    df      : our DataFrame
    returns : DataFrame with a new 'is_consistent' column
    """
    print("=" * 55)
    print("🔍 CHECKING ANNOTATION CONSISTENCY")
    print("=" * 55)
    
    # These are the 3 annotator columns in our CSV
    annotator_columns = ['annotator_1', 'annotator_2', 'annotator_3']
    
    # --- Normalise labels ---
    # Sometimes annotators write "bike" vs "bicycle" — same thing!
    # We create a mapping to standardise these variations
    label_mapping = {
        'bicycle': 'bike',   # treat bicycle and bike as same
        'human'  : 'person'  # treat human and person as same
    }
    
    # Apply the mapping to all annotator columns
    # .map() replaces values based on the dictionary
    # fillna keeps original value if no mapping exists
    for col in annotator_columns:
        df[col] = df[col].map(label_mapping).fillna(df[col])
    
    # --- Check consistency ---
    # nunique() counts how many UNIQUE values exist in a row
    # If nunique = 1 → all annotators agreed (consistent ✅)
    # If nunique > 1 → annotators disagreed (inconsistent ⚠️)
    
    # axis=1 means we check across columns (row by row)
    # We use dropna=False to count missing values too
    df['unique_labels'] = df[annotator_columns].nunique(axis=1, dropna=False)
    
    # is_consistent = True if all annotators gave same label
    df['is_consistent'] = df['unique_labels'] == 1
    
    # Count results
    consistent_count   = df['is_consistent'].sum()
    inconsistent_count = len(df) - consistent_count
    consistency_rate   = (consistent_count / len(df)) * 100
    
    print(f"✅ Consistent annotations   : {consistent_count} / {len(df)}")
    print(f"⚠️  Inconsistent annotations : {inconsistent_count} / {len(df)}")
    print(f"📊 Consistency Rate          : {consistency_rate:.1f}%\n")
    
    # Show the inconsistent rows so we can review them
    inconsistent_rows = df[df['is_consistent'] == False]
    if len(inconsistent_rows) > 0:
        print("⚠️  Inconsistent records (need review):")
        # Show only the relevant columns
        print(inconsistent_rows[['image_id'] + annotator_columns].to_string(index=False))
        print()
    
    return df, consistency_rate


# ============================================================
# STEP 4 — CALCULATE INTER-ANNOTATOR AGREEMENT (Cohen's Kappa)
# 
# Cohen's Kappa measures how much two annotators AGREE
# compared to random chance.
#
# Kappa Score Guide:
#   > 0.80  = Almost Perfect Agreement  ✅✅
#   0.60-0.80 = Substantial Agreement   ✅
#   0.40-0.60 = Moderate Agreement      ⚠️
#   < 0.40  = Poor Agreement            ❌
#
# In Amazon/ML annotation work, you want > 0.80
# ============================================================

def calculate_kappa(df):
    """
    Calculates Cohen's Kappa score between each pair of annotators.
    
    df      : our DataFrame
    returns : dictionary of kappa scores between annotator pairs
    """
    print("=" * 55)
    print("🔍 CALCULATING INTER-ANNOTATOR AGREEMENT (KAPPA)")
    print("=" * 55)
    
    annotator_columns = ['annotator_1', 'annotator_2', 'annotator_3']
    kappa_results = {}  # empty dictionary to store results
    
    # Compare every possible pair of annotators
    # Pairs: (1,2), (1,3), (2,3)
    pairs = [
        ('annotator_1', 'annotator_2'),
        ('annotator_1', 'annotator_3'),
        ('annotator_2', 'annotator_3')
    ]
    
    for pair in pairs:
        ann1, ann2 = pair  # unpack the pair names
        
        # We can only calculate kappa where BOTH annotators have labels
        # So we drop rows where either annotator has a missing value
        valid_rows = df[[ann1, ann2]].dropna()
        
        if len(valid_rows) < 2:
            print(f"⚠️  Not enough data to compare {ann1} vs {ann2}")
            continue
        
        # Calculate Cohen's Kappa using sklearn
        kappa = cohen_kappa_score(valid_rows[ann1], valid_rows[ann2])
        kappa_results[f"{ann1} vs {ann2}"] = round(kappa, 4)
        
        # Interpret the kappa score for easy reading
        if kappa > 0.80:
            interpretation = "✅✅ Almost Perfect"
        elif kappa > 0.60:
            interpretation = "✅  Substantial"
        elif kappa > 0.40:
            interpretation = "⚠️  Moderate"
        else:
            interpretation = "❌  Poor"
        
        print(f"  {ann1} vs {ann2}: Kappa = {kappa:.4f}  →  {interpretation}")
    
    # Calculate average kappa across all pairs
    if kappa_results:
        avg_kappa = sum(kappa_results.values()) / len(kappa_results)
        kappa_results['average'] = round(avg_kappa, 4)
        print(f"\n📊 Average Kappa Score: {avg_kappa:.4f}")
    
    print()
    return kappa_results


# ============================================================
# STEP 5 — FLAG LOW QUALITY ANNOTATIONS
# We flag records that have:
#   - Low confidence score (below 0.75)
#   - Slow annotation time (above 30 seconds — annotator confused)
#   - Missing labels
#   - Inconsistent labels between annotators
# ============================================================

def flag_low_quality(df):
    """
    Flags annotation records that fail quality thresholds.
    
    df      : our DataFrame
    returns : DataFrame with quality flag columns added
    """
    print("=" * 55)
    print("🔍 FLAGGING LOW QUALITY ANNOTATIONS")
    print("=" * 55)
    
    # --- Flag 1: Low confidence score ---
    # confidence_score < 0.75 means the annotator was not sure
    df['flag_low_confidence'] = df['confidence_score'] < 0.75
    
    # --- Flag 2: Slow annotation time ---
    # annotation_time_seconds > 30 means annotator took too long
    # This often means they were confused about the label
    df['flag_slow_annotation'] = df['annotation_time_seconds'] > 30
    
    # --- Flag 3: Missing labels ---
    # Check if ANY of the 3 annotators left a blank label
    annotator_columns = ['annotator_1', 'annotator_2', 'annotator_3']
    df['flag_missing_label'] = df[annotator_columns].isnull().any(axis=1)
    
    # --- Flag 4: Inconsistent labels (from Step 3) ---
    # We already calculated this in check_consistency()
    # is_consistent = False means inconsistent
    if 'is_consistent' in df.columns:
        df['flag_inconsistent'] = ~df['is_consistent']  # ~ means NOT
    else:
        df['flag_inconsistent'] = False
    
    # --- Overall Quality Flag ---
    # If ANY of the 4 flags above is True → needs review
    df['needs_review'] = (
        df['flag_low_confidence'] |      # | means OR
        df['flag_slow_annotation'] |
        df['flag_missing_label']   |
        df['flag_inconsistent']
    )
    
    # Count how many records need review
    needs_review_count = df['needs_review'].sum()
    good_quality_count = len(df) - needs_review_count
    quality_rate = (good_quality_count / len(df)) * 100
    
    print(f"✅ Good quality records  : {good_quality_count} / {len(df)}")
    print(f"🔴 Records needing review: {needs_review_count} / {len(df)}")
    print(f"📊 Overall Quality Rate  : {quality_rate:.1f}%\n")
    
    # Show breakdown of each flag type
    print("Flag Breakdown:")
    print(f"  Low confidence  : {df['flag_low_confidence'].sum()} records")
    print(f"  Slow annotation : {df['flag_slow_annotation'].sum()} records")
    print(f"  Missing labels  : {df['flag_missing_label'].sum()} records")
    print(f"  Inconsistent    : {df['flag_inconsistent'].sum()} records\n")
    
    return df, quality_rate


# ============================================================
# STEP 6 — GENERATE CHARTS / VISUALISATIONS
# Visual charts make it easy to see quality issues at a glance
# These are saved as PNG image files
# ============================================================

def generate_charts(df, output_folder):
    """
    Creates and saves quality visualisation charts.
    
    df            : our DataFrame with all quality flags
    output_folder : folder where charts will be saved
    """
    print("=" * 55)
    print("📊 GENERATING QUALITY CHARTS")
    print("=" * 55)
    
    # Set chart style — makes charts look professional
    sns.set_style("whitegrid")
    
    # Create a figure with 2 charts side by side
    # figsize = width x height in inches
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Annotation Quality Dashboard', fontsize=16, fontweight='bold')
    
    # --- Chart 1: Quality Status Pie Chart ---
    # Count how many records are Good vs Need Review
    quality_counts = df['needs_review'].value_counts()
    labels = ['Good Quality', 'Needs Review']
    colors = ['#2ecc71', '#e74c3c']  # green and red
    
    axes[0].pie(
        quality_counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',  # show percentage on chart
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[0].set_title('Overall Annotation Quality', fontsize=13)
    
    # --- Chart 2: Flag Breakdown Bar Chart ---
    flag_columns = [
        'flag_low_confidence',
        'flag_slow_annotation',
        'flag_missing_label',
        'flag_inconsistent'
    ]
    flag_labels = [
        'Low\nConfidence',
        'Slow\nAnnotation',
        'Missing\nLabels',
        'Inconsistent\nLabels'
    ]
    # Count True values in each flag column
    flag_counts = [df[col].sum() for col in flag_columns]
    
    bars = axes[1].bar(flag_labels, flag_counts, color=['#e74c3c','#e67e22','#9b59b6','#3498db'])
    axes[1].set_title('Quality Issues by Type', fontsize=13)
    axes[1].set_ylabel('Number of Records')
    axes[1].set_ylim(0, max(flag_counts) + 2)
    
    # Add count numbers on top of each bar
    for bar, count in zip(bars, flag_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.1,
            str(count),
            ha='center', va='bottom', fontweight='bold'
        )
    
    plt.tight_layout()
    
    # Save the chart as a PNG file
    chart_path = os.path.join(output_folder, 'quality_dashboard.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()  # close the chart to free memory
    
    print(f"✅ Chart saved to: {chart_path}\n")
    
    return chart_path


# ============================================================
# STEP 7 — GENERATE QUALITY REPORT (CSV)
# Save all results to a CSV file that can be shared
# with your team or manager
# ============================================================

def generate_report(df, kappa_results, consistency_rate, quality_rate, output_folder):
    """
    Generates and saves the final quality report as CSV files.
    
    df                : our DataFrame with all quality data
    kappa_results     : dictionary of kappa scores
    consistency_rate  : percentage of consistent annotations
    quality_rate      : overall quality percentage
    output_folder     : folder where reports will be saved
    """
    print("=" * 55)
    print("📝 GENERATING QUALITY REPORT")
    print("=" * 55)
    
    # --- Report 1: Full annotation data with all flags ---
    full_report_path = os.path.join(output_folder, 'quality_report_full.csv')
    df.to_csv(full_report_path, index=False)
    print(f"✅ Full report saved to    : {full_report_path}")
    
    # --- Report 2: Only records that need review ---
    review_records = df[df['needs_review'] == True]
    review_report_path = os.path.join(output_folder, 'records_needing_review.csv')
    review_records.to_csv(review_report_path, index=False)
    print(f"✅ Review report saved to  : {review_report_path}")
    
    # --- Report 3: Summary statistics ---
    summary = {
        'Report Generated At'       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Total Records Checked'     : len(df),
        'Good Quality Records'      : int((len(df) * quality_rate) / 100),
        'Records Needing Review'    : int(df['needs_review'].sum()),
        'Consistency Rate (%)'      : f"{consistency_rate:.1f}%",
        'Overall Quality Rate (%)'  : f"{quality_rate:.1f}%",
        'Avg Kappa Score'           : kappa_results.get('average', 'N/A'),
        'Low Confidence Count'      : int(df['flag_low_confidence'].sum()),
        'Slow Annotation Count'     : int(df['flag_slow_annotation'].sum()),
        'Missing Label Count'       : int(df['flag_missing_label'].sum()),
        'Inconsistent Label Count'  : int(df['flag_inconsistent'].sum()),
    }
    
    # Convert summary dict to DataFrame for saving
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    summary_path = os.path.join(output_folder, 'quality_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary report saved to : {summary_path}\n")
    
    # Print summary to console as well
    print("=" * 55)
    print("📋 FINAL QUALITY SUMMARY")
    print("=" * 55)
    for metric, value in summary.items():
        print(f"  {metric:<35}: {value}")
    print()
    
    return summary


# ============================================================
# MAIN FUNCTION — Runs everything in order
# This is the entry point of the program.
# When you run this file, Python starts here.
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("  ANNOTATION QUALITY CHECKER — by Arunashree S")
    print("=" * 55)
    
    # --- Define file paths ---
    # os.path.dirname gets the folder of this script
    # os.path.join combines folder paths safely
    base_folder   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path     = os.path.join(base_folder, 'data', 'sample_annotations.csv')
    output_folder = os.path.join(base_folder, 'output')
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # -------------------------------------------------------
    # RUN ALL STEPS IN ORDER
    # -------------------------------------------------------
    
    # Step 1: Load the data
    df = load_data(data_path)
    
    # Step 2: Check for missing values
    missing_summary = check_missing_values(df)
    
    # Step 3: Check consistency between annotators
    df, consistency_rate = check_consistency(df)
    
    # Step 4: Calculate Cohen's Kappa (inter-annotator agreement)
    kappa_results = calculate_kappa(df)
    
    # Step 5: Flag low quality records
    df, quality_rate = flag_low_quality(df)
    
    # Step 6: Generate visual charts
    chart_path = generate_charts(df, output_folder)
    
    # Step 7: Generate and save reports
    summary = generate_report(df, kappa_results, consistency_rate, quality_rate, output_folder)
    
    print("=" * 55)
    print("✅ QUALITY CHECK COMPLETE!")
    print(f"   Output files saved to: {output_folder}")
    print("=" * 55 + "\n")


# This line means: only run main() if this file is run directly
# (not when it's imported by another file)
if __name__ == "__main__":
    main()
