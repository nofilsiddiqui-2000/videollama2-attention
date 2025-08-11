#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_results(file="sample_results.txt"):
    """Detailed analysis of FGSM results per video, then summary + CSV export"""
    
    try:
        df = pd.read_csv(file, sep='\t')
        
        print(f"\n📊 FGSM Attack Results - Detailed Per Video")
        print("=" * 80)
        print(f"Total videos processed: {len(df)}")
        print()
        
        # Add derived columns for analysis
        df['Attack_Success'] = df['BERTScore_F1'] < 0.6
        df['Strong_Attack'] = df['BERTScore_F1'] < 0.5
        df['CLIPScore_Change'] = df['Adversarial_CLIPScore'] - df['Original_CLIPScore']
        df['CLIPScore_Degraded'] = df['CLIPScore_Change'] < 0
        df['Attack_Quality'] = df['BERTScore_F1'].apply(lambda x: 
            'Excellent' if x < 0.5 else 
            'Good' if x < 0.6 else 
            'Weak' if x < 0.7 else 'Failed')
        
        # Extract action category from filename (if available)
        df['Action_Category'] = df['Video_Filename'].str.extract(r'^([^_]+)')[0].fillna('Unknown')
        
        # Per-video results
        print("📹 Per-Video Results:")
        print("-" * 80)
        
        for idx, row in df.iterrows():
            video_name = row['Video_Filename']
            bert_f1 = row['BERTScore_F1']
            sbert_sim = row['SBERT_Sim']
            feat_sim = row['Feature_CosSim']
            orig_clip = row['Original_CLIPScore']
            adv_clip = row['Adversarial_CLIPScore']
            psnr = row['PSNR_dB']
            linf_norm = row['Linf_Norm']
            proc_time = row['Processing_Time_Sec']
            
            # Determine attack success
            attack_success = "🔥 SUCCESS" if bert_f1 < 0.6 else "⚠️  WEAK"
            clip_change = adv_clip - orig_clip
            
            print(f"Video {idx+1}: {video_name}")
            print(f"  BERTScore F1:        {bert_f1:.3f} {attack_success}")
            print(f"  SBERT Similarity:    {sbert_sim:.3f}")
            print(f"  Feature Similarity:  {feat_sim:.3f}")
            print(f"  CLIPScore Change:    {orig_clip:.2f} → {adv_clip:.2f} ({clip_change:+.2f})")
            print(f"  PSNR:               {psnr:.2f} dB")
            print(f"  L∞ Norm:            {linf_norm:.4f}")
            print(f"  Processing Time:     {proc_time:.1f}s")
            print()
        
        print("=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("=" * 80)
        
        # Calculate averages and statistics
        stats = calculate_summary_stats(df)
        display_summary_stats(stats, df)
        
        # Generate CSV files
        generate_csv_files(df, stats, file)
        
        print("\n" + "=" * 80)
        
        return df, stats
        
    except FileNotFoundError:
        print(f"❌ File not found: {file}")
        print("💡 Run FGSM evaluation first:")
        print("   python fgsm_batch.py sample_dataset --epsilon 0.05 --caption-file sample_results.txt")
        return None, None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None

def calculate_summary_stats(df):
    """Calculate comprehensive summary statistics"""
    
    stats = {
        # Basic metrics
        'avg_bert': df['BERTScore_F1'].mean(),
        'std_bert': df['BERTScore_F1'].std(),
        'avg_sbert': df['SBERT_Sim'].mean(),
        'std_sbert': df['SBERT_Sim'].std(),
        'avg_feat': df['Feature_CosSim'].mean(),
        'std_feat': df['Feature_CosSim'].std(),
        'avg_orig_clip': df['Original_CLIPScore'].mean(),
        'std_orig_clip': df['Original_CLIPScore'].std(),
        'avg_adv_clip': df['Adversarial_CLIPScore'].mean(),
        'std_adv_clip': df['Adversarial_CLIPScore'].std(),
        'avg_psnr': df['PSNR_dB'].mean(),
        'std_psnr': df['PSNR_dB'].std(),
        'avg_linf': df['Linf_Norm'].mean(),
        'std_linf': df['Linf_Norm'].std(),
        'avg_time': df['Processing_Time_Sec'].mean(),
        'total_time': df['Processing_Time_Sec'].sum(),
        
        # Attack effectiveness
        'successful_attacks': df['Attack_Success'].sum(),
        'success_rate': df['Attack_Success'].mean() * 100,
        'strong_attacks': df['Strong_Attack'].sum(),
        'strong_rate': df['Strong_Attack'].mean() * 100,
        'clip_degraded': df['CLIPScore_Degraded'].sum(),
        'clip_degraded_rate': df['CLIPScore_Degraded'].mean() * 100,
        
        # Best/worst performers
        'best_attack_idx': df['BERTScore_F1'].idxmin(),
        'worst_attack_idx': df['BERTScore_F1'].idxmax(),
        
        # Additional metrics
        'total_videos': len(df),
        'unique_categories': df['Action_Category'].nunique(),
        'avg_clip_change': df['CLIPScore_Change'].mean(),
    }
    
    return stats

def display_summary_stats(stats, df):
    """Display summary statistics"""
    
    print(f"BERTScore F1:          {stats['avg_bert']:.3f} ± {stats['std_bert']:.3f}")
    print(f"SBERT Similarity:      {stats['avg_sbert']:.3f} ± {stats['std_sbert']:.3f}")
    print(f"Feature Similarity:    {stats['avg_feat']:.3f} ± {stats['std_feat']:.3f}")
    print(f"Original CLIPScore:    {stats['avg_orig_clip']:.2f} ± {stats['std_orig_clip']:.2f}")
    print(f"Adversarial CLIPScore: {stats['avg_adv_clip']:.2f} ± {stats['std_adv_clip']:.2f}")
    print(f"CLIPScore Change:      {stats['avg_clip_change']:+.2f}")
    print(f"PSNR (dB):            {stats['avg_psnr']:.2f} ± {stats['std_psnr']:.2f}")
    print(f"L∞ Norm:              {stats['avg_linf']:.4f} ± {stats['std_linf']:.4f}")
    print(f"Processing Time:       {stats['avg_time']:.1f}s per video")
    print(f"Total Time:           {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} min)")
    
    print("\n🎯 ATTACK EFFECTIVENESS:")
    print("-" * 40)
    print(f"Successful Attacks (BERTScore < 0.6): {stats['successful_attacks']}/{stats['total_videos']} ({stats['success_rate']:.1f}%)")
    print(f"Strong Attacks (BERTScore < 0.5):     {stats['strong_attacks']}/{stats['total_videos']} ({stats['strong_rate']:.1f}%)")
    print(f"CLIPScore Degradation:                {stats['clip_degraded']}/{stats['total_videos']} ({stats['clip_degraded_rate']:.1f}%)")
    
    # Best and worst performers
    best_video = df.loc[stats['best_attack_idx']]
    worst_video = df.loc[stats['worst_attack_idx']]
    
    print(f"\n🏆 BEST ATTACK (Lowest BERTScore F1):")
    print(f"   Video: {best_video['Video_Filename']}")
    print(f"   BERTScore F1: {best_video['BERTScore_F1']:.3f}")
    print(f"   Feature Sim: {best_video['Feature_CosSim']:.3f}")
    print(f"   CLIPScore: {best_video['Original_CLIPScore']:.2f} → {best_video['Adversarial_CLIPScore']:.2f}")
    
    print(f"\n🔻 WEAKEST ATTACK (Highest BERTScore F1):")
    print(f"   Video: {worst_video['Video_Filename']}")
    print(f"   BERTScore F1: {worst_video['BERTScore_F1']:.3f}")
    print(f"   Feature Sim: {worst_video['Feature_CosSim']:.3f}")
    print(f"   CLIPScore: {worst_video['Original_CLIPScore']:.2f} → {worst_video['Adversarial_CLIPScore']:.2f}")

def generate_csv_files(df, stats, original_file):
    """Generate comprehensive CSV files for analysis"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = original_file.replace('.txt', '').replace('.tsv', '')
    
    print(f"\n📊 GENERATING CSV FILES:")
    print("-" * 40)
    
    # 1. Enhanced per-video results
    detailed_df = df.copy()
    detailed_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detailed_df['User'] = 'nofilsiddiqui-2000'
    detailed_df['Epsilon'] = 0.05  # Add epsilon value
    
    detailed_csv = f"{base_name}_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✅ Detailed results: {detailed_csv}")
    
    # 2. Summary statistics CSV
    summary_data = {
        'Metric': [
            'Total Videos', 'Unique Categories', 'Success Rate (%)', 'Strong Attack Rate (%)',
            'CLIPScore Degradation Rate (%)', 'Avg BERTScore F1', 'Std BERTScore F1',
            'Avg SBERT Similarity', 'Std SBERT Similarity', 'Avg Feature Similarity',
            'Std Feature Similarity', 'Avg Original CLIPScore', 'Std Original CLIPScore',
            'Avg Adversarial CLIPScore', 'Std Adversarial CLIPScore', 'Avg CLIPScore Change',
            'Avg PSNR (dB)', 'Std PSNR (dB)', 'Avg L∞ Norm', 'Std L∞ Norm',
            'Avg Processing Time (s)', 'Total Processing Time (s)', 'Total Processing Time (min)'
        ],
        'Value': [
            stats['total_videos'], stats['unique_categories'], stats['success_rate'], stats['strong_rate'],
            stats['clip_degraded_rate'], stats['avg_bert'], stats['std_bert'],
            stats['avg_sbert'], stats['std_sbert'], stats['avg_feat'],
            stats['std_feat'], stats['avg_orig_clip'], stats['std_orig_clip'],
            stats['avg_adv_clip'], stats['std_adv_clip'], stats['avg_clip_change'],
            stats['avg_psnr'], stats['std_psnr'], stats['avg_linf'], stats['std_linf'],
            stats['avg_time'], stats['total_time'], stats['total_time']/60
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df['User'] = 'nofilsiddiqui-2000'
    summary_df['Dataset'] = base_name
    
    summary_csv = f"{base_name}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary statistics: {summary_csv}")
    
    # 3. Category-wise analysis (if categories detected)
    if stats['unique_categories'] > 1:
        category_stats = df.groupby('Action_Category').agg({
            'BERTScore_F1': ['count', 'mean', 'std', 'min', 'max'],
            'Feature_CosSim': ['mean', 'std'],
            'CLIPScore_Change': ['mean', 'std'],
            'Attack_Success': 'sum',
            'Strong_Attack': 'sum',
            'Processing_Time_Sec': 'mean'
        }).round(4)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.reset_index()
        category_stats['Success_Rate'] = (category_stats['Attack_Success_sum'] / category_stats['BERTScore_F1_count'] * 100).round(1)
        category_stats['Strong_Rate'] = (category_stats['Strong_Attack_sum'] / category_stats['BERTScore_F1_count'] * 100).round(1)
        
        category_csv = f"{base_name}_by_category_{timestamp}.csv"
        category_stats.to_csv(category_csv, index=False)
        print(f"✅ Category analysis: {category_csv}")
    
    # 4. Attack effectiveness ranking
    ranking_df = df[['Video_Filename', 'Action_Category', 'BERTScore_F1', 'Feature_CosSim', 
                     'CLIPScore_Change', 'Attack_Quality', 'PSNR_dB']].copy()
    ranking_df = ranking_df.sort_values('BERTScore_F1')
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[['Rank'] + [col for col in ranking_df.columns if col != 'Rank']]
    
    ranking_csv = f"{base_name}_ranking_{timestamp}.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    print(f"✅ Attack ranking: {ranking_csv}")
    
    print(f"\n📁 All CSV files saved with timestamp: {timestamp}")
    print(f"🔍 Use these files for further analysis in Excel, R, or Python")

if __name__ == "__main__":
    import sys
    
    # Allow custom file as argument
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "sample_results.txt"
    
    df, stats = analyze_results(results_file)
    
    if df is not None:
        print(f"\n✅ Analysis complete! Check the generated CSV files for detailed data.")
