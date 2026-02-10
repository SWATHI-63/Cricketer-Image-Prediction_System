"""
Data Preprocessing Script for Cricketer Image Prediction System
This script processes the players.csv file and prepares the dataset for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# ===================================
# CONFIGURATION
# ===================================

DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "players.csv"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "processed"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================
# LOAD AND EXPLORE DATA
# ===================================

def load_data():
    """Load the players dataset"""
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset loaded: {len(df)} records")
    return df

def explore_data(df):
    """Display basic information about the dataset"""
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    print(f"\nTotal records: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

# ===================================
# CLEAN DATA
# ===================================

def clean_data(df):
    """Clean and process the dataset"""
    print("\n" + "="*50)
    print("CLEANING DATA")
    print("="*50)
    
    # Remove the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✓ Removed unnamed index column")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"✓ Found {missing_count} missing values")
        df = df.dropna()
        print(f"✓ Removed rows with missing values. Remaining: {len(df)}")
    else:
        print("✓ No missing values found")
    
    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"✓ Found {duplicates} duplicate rows")
        df = df.drop_duplicates()
        print(f"✓ Removed duplicates. Remaining: {len(df)}")
    else:
        print("✓ No duplicate rows found")
    
    return df

# ===================================
# PROCESS PLAYER NAMES
# ===================================

def process_player_names(df):
    """Standardize player names"""
    print("\n" + "="*50)
    print("PROCESSING PLAYER NAMES")
    print("="*50)
    
    # Store original names
    df['original_player'] = df['player']
    
    # Clean player names: replace underscores with spaces and title case
    df['player_clean'] = df['player'].str.replace('_', ' ').str.title()
    
    # Replace special abbreviations
    df['player_clean'] = df['player_clean'].str.replace('K. L. Rahul', 'KL Rahul')
    
    print("✓ Player names standardized")
    print(f"\nUnique players: {df['player'].nunique()}")
    print("\nPlayer distribution:")
    print(df['player_clean'].value_counts())
    
    return df

# ===================================
# VERIFY IMAGE FILES
# ===================================

def verify_images(df):
    """Check if all referenced images exist"""
    print("\n" + "="*50)
    print("VERIFYING IMAGE FILES")
    print("="*50)
    
    missing_images = []
    existing_images = []
    
    for idx, row in df.iterrows():
        image_path = IMAGES_DIR / row['image']
        if image_path.exists():
            existing_images.append(row['image'])
        else:
            missing_images.append(row['image'])
    
    print(f"✓ Total images referenced: {len(df)}")
    print(f"✓ Images found: {len(existing_images)}")
    print(f"✓ Images missing: {len(missing_images)}")
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images are missing!")
        print(f"First 5 missing: {missing_images[:5]}")
        
        # Remove rows with missing images
        df['image_exists'] = df['image'].apply(lambda x: (IMAGES_DIR / x).exists())
        df = df[df['image_exists'] == True].drop('image_exists', axis=1)
        print(f"✓ Removed rows with missing images. Remaining: {len(df)}")
    
    return df

# ===================================
# CREATE TRAIN/TEST SPLIT METADATA
# ===================================

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train/test split metadata"""
    print("\n" + "="*50)
    print("CREATING TRAIN/TEST SPLIT")
    print("="*50)
    
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class distribution
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['player'],
        random_state=random_state
    )
    
    print(f"✓ Training set: {len(train_df)} images")
    print(f"✓ Test set: {len(test_df)} images")
    
    print("\nClass distribution in training set:")
    print(train_df['player_clean'].value_counts())
    
    return train_df, test_df

# ===================================
# SAVE PROCESSED DATA
# ===================================

def save_processed_data(train_df, test_df, full_df):
    """Save processed datasets to CSV files"""
    print("\n" + "="*50)
    print("SAVING PROCESSED DATA")
    print("="*50)
    
    # Save full processed dataset
    full_output = OUTPUT_DIR / "players_processed.csv"
    full_df.to_csv(full_output, index=False)
    print(f"✓ Saved full dataset: {full_output}")
    
    # Save train split
    train_output = OUTPUT_DIR / "train_data.csv"
    train_df.to_csv(train_output, index=False)
    print(f"✓ Saved training data: {train_output}")
    
    # Save test split
    test_output = OUTPUT_DIR / "test_data.csv"
    test_df.to_csv(test_output, index=False)
    print(f"✓ Saved test data: {test_output}")
    
    # Create label mapping
    label_mapping = {name: idx for idx, name in enumerate(sorted(full_df['player'].unique()))}
    label_df = pd.DataFrame(list(label_mapping.items()), columns=['player', 'label_id'])
    label_df['player_clean'] = label_df['player'].str.replace('_', ' ').str.title()
    
    label_output = OUTPUT_DIR / "label_mapping.csv"
    label_df.to_csv(label_output, index=False)
    print(f"✓ Saved label mapping: {label_output}")
    
    return full_output, train_output, test_output, label_output

# ===================================
# GENERATE SUMMARY STATISTICS
# ===================================

def generate_summary(df, train_df, test_df):
    """Generate and save summary statistics"""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    summary = {
        'Total Images': len(df),
        'Training Images': len(train_df),
        'Test Images': len(test_df),
        'Number of Players': df['player'].nunique(),
        'Average Images per Player': round(len(df) / df['player'].nunique(), 2),
        'Min Images per Player': df['player'].value_counts().min(),
        'Max Images per Player': df['player'].value_counts().max()
    }
    
    print("\nDataset Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_output = OUTPUT_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"\n✓ Saved summary: {summary_output}")
    
    # Player-wise statistics
    player_stats = df.groupby('player_clean').agg({
        'image': 'count'
    }).rename(columns={'image': 'image_count'}).sort_values('image_count', ascending=False)
    
    player_stats_output = OUTPUT_DIR / "player_statistics.csv"
    player_stats.to_csv(player_stats_output)
    print(f"✓ Saved player statistics: {player_stats_output}")

# ===================================
# MAIN EXECUTION
# ===================================

def main():
    """Main execution function"""
    print("\n" + "="*50)
    print("CRICKETER IMAGE PREDICTION SYSTEM")
    print("DATA PREPROCESSING SCRIPT")
    print("="*50)
    
    try:
        # Load data
        df = load_data()
        
        # Explore data
        df = explore_data(df)
        
        # Clean data
        df = clean_data(df)
        
        # Process player names
        df = process_player_names(df)
        
        # Verify images
        df = verify_images(df)
        
        # Create train/test split
        train_df, test_df = create_train_test_split(df)
        
        # Save processed data
        save_processed_data(train_df, test_df, df)
        
        # Generate summary
        generate_summary(df, train_df, test_df)
        
        print("\n" + "="*50)
        print("✓ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"\nProcessed files saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
