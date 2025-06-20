#!/usr/bin/env python3
import pandas as pd

def analyze_results(file="sample_results.txt"):
    """Simple analysis of FGSM results"""
    
    try:
        df = pd.read_csv(file, sep='\t')
        print(f"\n📊 FGSM Attack Results Summary")
        print("=" * 40)
        print(f"Videos processed: {len(df)}")
        print(f"Average BERTScore F1: {df['BERTScore_F1'].mean():.3f}")
        print(f"Average Feature Similarity: {df['Feature_CosSim'].mean():.3f}")
        print(f"Average CLIPScore (Original): {df['Original_CLIPScore'].mean():.2f}")
        print(f"Average CLIPScore (Adversarial): {df['Adversarial_CLIPScore'].mean():.2f}")
        print(f"Average Processing Time: {df['Processing_Time_Sec'].mean():.1f}s")
        
        # Success rate (lower BERTScore = better attack)
        successful = (df['BERTScore_F1'] < 0.6).sum()
        print(f"Successful attacks: {successful}/{len(df)} ({successful/len(df)*100:.1f}%)")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file}")
        print("Run FGSM evaluation first!")

if __name__ == "__main__":
    analyze_results()
