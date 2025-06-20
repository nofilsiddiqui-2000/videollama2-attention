#!/usr/bin/env python3
import pandas as pd

def analyze_results(file="sample_results.txt"):
    """Detailed analysis of FGSM results per video, then summary"""
    
    try:
        df = pd.read_csv(file, sep='\t')
        
        print(f"\n📊 FGSM Attack Results - Detailed Per Video")
        print("=" * 80)
        print(f"Total videos processed: {len(df)}")
        print()
        
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
        avg_bert = df['BERTScore_F1'].mean()
        std_bert = df['BERTScore_F1'].std()
        avg_sbert = df['SBERT_Sim'].mean()
        std_sbert = df['SBERT_Sim'].std()
        avg_feat = df['Feature_CosSim'].mean()
        std_feat = df['Feature_CosSim'].std()
        avg_orig_clip = df['Original_CLIPScore'].mean()
        std_orig_clip = df['Original_CLIPScore'].std()
        avg_adv_clip = df['Adversarial_CLIPScore'].mean()
        std_adv_clip = df['Adversarial_CLIPScore'].std()
        avg_psnr = df['PSNR_dB'].mean()
        std_psnr = df['PSNR_dB'].std()
        avg_linf = df['Linf_Norm'].mean()
        std_linf = df['Linf_Norm'].std()
        avg_time = df['Processing_Time_Sec'].mean()
        total_time = df['Processing_Time_Sec'].sum()
        
        print(f"BERTScore F1:          {avg_bert:.3f} ± {std_bert:.3f}")
        print(f"SBERT Similarity:      {avg_sbert:.3f} ± {std_sbert:.3f}")
        print(f"Feature Similarity:    {avg_feat:.3f} ± {std_feat:.3f}")
        print(f"Original CLIPScore:    {avg_orig_clip:.2f} ± {std_orig_clip:.2f}")
        print(f"Adversarial CLIPScore: {avg_adv_clip:.2f} ± {std_adv_clip:.2f}")
        print(f"CLIPScore Change:      {avg_adv_clip - avg_orig_clip:+.2f}")
        print(f"PSNR (dB):            {avg_psnr:.2f} ± {std_psnr:.2f}")
        print(f"L∞ Norm:              {avg_linf:.4f} ± {std_linf:.4f}")
        print(f"Processing Time:       {avg_time:.1f}s per video")
        print(f"Total Time:           {total_time:.1f}s ({total_time/60:.1f} min)")
        
        print("\n🎯 ATTACK EFFECTIVENESS:")
        print("-" * 40)
        
        # Success metrics
        successful_attacks = (df['BERTScore_F1'] < 0.6).sum()
        success_rate = successful_attacks / len(df) * 100
        
        strong_attacks = (df['BERTScore_F1'] < 0.5).sum()
        strong_rate = strong_attacks / len(df) * 100
        
        clip_degraded = (df['Adversarial_CLIPScore'] < df['Original_CLIPScore']).sum()
        clip_degraded_rate = clip_degraded / len(df) * 100
        
        print(f"Successful Attacks (BERTScore < 0.6): {successful_attacks}/{len(df)} ({success_rate:.1f}%)")
        print(f"Strong Attacks (BERTScore < 0.5):     {strong_attacks}/{len(df)} ({strong_rate:.1f}%)")
        print(f"CLIPScore Degradation:                {clip_degraded}/{len(df)} ({clip_degraded_rate:.1f}%)")
        
        # Best and worst performers
        print(f"\n🏆 BEST ATTACK (Lowest BERTScore F1):")
        best_idx = df['BERTScore_F1'].idxmin()
        best_video = df.loc[best_idx]
        print(f"   Video: {best_video['Video_Filename']}")
        print(f"   BERTScore F1: {best_video['BERTScore_F1']:.3f}")
        print(f"   Feature Sim: {best_video['Feature_CosSim']:.3f}")
        print(f"   CLIPScore: {best_video['Original_CLIPScore']:.2f} → {best_video['Adversarial_CLIPScore']:.2f}")
        
        print(f"\n🔻 WEAKEST ATTACK (Highest BERTScore F1):")
        worst_idx = df['BERTScore_F1'].idxmax()
        worst_video = df.loc[worst_idx]
        print(f"   Video: {worst_video['Video_Filename']}")
        print(f"   BERTScore F1: {worst_video['BERTScore_F1']:.3f}")
        print(f"   Feature Sim: {worst_video['Feature_CosSim']:.3f}")
        print(f"   CLIPScore: {worst_video['Original_CLIPScore']:.2f} → {worst_video['Adversarial_CLIPScore']:.2f}")
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file}")
        print("💡 Run FGSM evaluation first:")
        print("   python fgsm_batch.py sample_dataset --epsilon 0.05 --caption-file sample_results.txt")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    import sys
    
    # Allow custom file as argument
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "sample_results.txt"
    
    analyze_results(results_file)
