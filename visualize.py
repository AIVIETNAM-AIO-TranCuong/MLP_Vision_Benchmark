import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    results_dir = "results"
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    
    results = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
            
        cfg = data.get("config", {})
        results.append({
            "Model": cfg.get("model", "unknown").upper(),
            "Size": cfg.get("size", "unknown").capitalize(),
            "Params": cfg.get("n_params", 0) / 1e6, # in Millions
            "TestAcc": data.get("test_acc", 0.0)
        })
        
    df = pd.DataFrame(results)
    
    # Define a clean style
    sns.set_theme(style="whitegrid")
    
    # 1. Bar Chart: Accuracy comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="Model", y="TestAcc", hue="Size", palette="viridis")
    plt.ylim(60, 95)
    plt.title("So sánh Test Accuracy giữa các Model và Kích thước trên CIFAR-10", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Test Accuracy (%)", fontweight='bold')
    plt.xlabel("Mô hình", fontweight='bold')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=9)
                   
    plt.legend(title='Kích thước (Size)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Scatter Plot: Params vs Accuracy
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x="Params", y="TestAcc", hue="Model", style="Size", s=200, palette="Set1")
    plt.title("Sự đánh đổi: Số lượng tham số (Params) vs Độ chính xác (Accuracy)", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Số tham số (Triệu - Millions)", fontweight='bold')
    plt.ylabel("Test Accuracy (%)", fontweight='bold')
    
    # Annotate points
    for i in range(df.shape[0]):
        plt.text(x=df.Params[i] + 0.2, y=df.TestAcc[i], s=df.Size[i], 
                 fontdict=dict(color='gray', size=9))
                 
    plt.legend(title='Mô hình', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "params_vs_accuracy.png"), dpi=300)
    plt.close()

    # 3. Line Chart: Learning Curve (Val Acc over Epochs for Small models)
    plt.figure(figsize=(12, 6))
    
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        
        # Only plot learning curves for "Small" models to keep the chart clean
        if cfg.get("size") == "small":
            model_name = cfg.get("model", "unknown").upper()
            epochs_data = data.get("epochs", [])
            
            # Extract epoch numbers and validation accuracies
            epoch_nums = [e.get("epoch") for e in epochs_data]
            val_accs = [e.get("val_acc") for e in epochs_data]
            
            if epoch_nums and val_accs:
                sns.lineplot(x=epoch_nums, y=val_accs, label=model_name, linewidth=2.5)

    plt.title("Đường cong học tập (Learning Curve) - Các bản kích thước Small", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Validation Accuracy (%)", fontweight='bold')
    plt.legend(title='Mô hình', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve_small.png"), dpi=300)
    plt.close()
    
    print("Saved visualizations to visualizations/ folder.")

if __name__ == "__main__":
    main()
