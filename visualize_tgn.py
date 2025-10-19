"""
TGN Interpretability Visualizer

Visualize what TGN "sees" at each stage:
1. Input image
2. ResNet trunk features
3. Gating network predictions (superclass probabilities)
4. Expert network activations
5. Final predictions

Usage:
    python visualize_tgn.py --checkpoint checkpoints_tgn/tgn_best.pt
    python visualize_tgn.py --checkpoint checkpoints_tgn/tgn_best.pt --image-idx 42
    python visualize_tgn.py --checkpoint checkpoints_tgn/tgn_best.pt --compare-errors
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

from tgn import TaxonomicGatedNetwork, make_dataloaders, COARSE_FINE_LABELS, fine_to_coarse

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

CIFAR100_SUPERCLASSES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects',
    'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
    'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]


class TGNVisualizer:
    """Extract and visualize intermediate activations from TGN"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.activations = {}
        self.hooks = []
        
        # Register hooks to capture intermediate features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook into ResNet-18 layers
        self.hooks.append(
            self.model.trunk.backbone[4].register_forward_hook(get_activation('layer1'))
        )
        self.hooks.append(
            self.model.trunk.backbone[5].register_forward_hook(get_activation('layer2'))
        )
        self.hooks.append(
            self.model.trunk.backbone[6].register_forward_hook(get_activation('layer3'))
        )
        self.hooks.append(
            self.model.trunk.backbone[7].register_forward_hook(get_activation('layer4'))
        )
        
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def analyze_image(self, image, fine_label, coarse_label):
        """
        Run image through model and extract all intermediate features.
        
        Returns:
            dict with keys: 'features', 'gating_logits', 'gating_probs', 
                           'expert_outputs', 'final_probs', 'prediction'
        """
        self.activations = {}
        
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            
            # Forward pass
            final_probs, gating_logits, expert_outputs = self.model(image)
            
            # Compute probabilities
            gating_probs = F.softmax(gating_logits, dim=1).squeeze(0)
            final_probs_exp = torch.exp(final_probs.squeeze(0))  # Convert log probs
            
            # Get prediction
            pred_class = final_probs.argmax(dim=1).item()
            pred_confidence = final_probs_exp[pred_class].item()
            
            # Get top-3 predictions
            top3_probs, top3_classes = torch.topk(final_probs_exp, k=3)
            
            return {
                'layer_features': {
                    'layer1': self.activations.get('layer1'),
                    'layer2': self.activations.get('layer2'),
                    'layer3': self.activations.get('layer3'),
                    'layer4': self.activations.get('layer4'),
                },
                'gating_logits': gating_logits.squeeze(0).cpu(),
                'gating_probs': gating_probs.cpu(),
                'expert_outputs': expert_outputs.squeeze(0).cpu(),
                'final_probs': final_probs_exp.cpu(),
                'prediction': pred_class,
                'confidence': pred_confidence,
                'top3_classes': top3_classes.cpu().numpy(),
                'top3_probs': top3_probs.cpu().numpy(),
                'true_fine': fine_label,
                'true_coarse': coarse_label,
            }


def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image"""
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_feature_maps(layer_features, layer_name, n_channels=8):
    """Visualize first N channels of a feature map"""
    features = layer_features.squeeze(0).cpu().numpy()
    n_channels = min(n_channels, features.shape[0])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'{layer_name} Feature Maps (first {n_channels} channels)', fontsize=14)
    
    for i in range(n_channels):
        ax = axes[i // 4, i % 4]
        feature_map = features[i]
        
        # Normalize for display
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    return fig


def visualize_single_image(visualizer, image, fine_label, coarse_label, save_path=None):
    """Complete visualization for a single image"""
    
    results = visualizer.analyze_image(image, fine_label, coarse_label)
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Original Image
    ax_img = fig.add_subplot(gs[0, 0])
    img_display = denormalize_image(image)
    ax_img.imshow(img_display)
    ax_img.set_title('Input Image', fontsize=14, fontweight='bold')
    ax_img.axis('off')
    
    # Add ground truth labels
    true_fine_name = CIFAR100_CLASSES[fine_label]
    true_coarse_name = CIFAR100_SUPERCLASSES[coarse_label]
    ax_img.text(0.5, -0.15, f'True: {true_fine_name}\n({true_coarse_name})', 
                transform=ax_img.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 2. Prediction Info
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.axis('off')
    
    pred_name = CIFAR100_CLASSES[results['prediction']]
    pred_coarse = fine_to_coarse(torch.tensor(results['prediction'])).item()
    pred_coarse_name = CIFAR100_SUPERCLASSES[pred_coarse]
    
    is_correct = results['prediction'] == fine_label
    color = 'green' if is_correct else 'red'
    
    pred_text = f"Prediction: {pred_name}\n"
    pred_text += f"Superclass: {pred_coarse_name}\n"
    pred_text += f"Confidence: {results['confidence']:.2%}\n"
    pred_text += f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}"
    
    ax_pred.text(0.1, 0.5, pred_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    ax_pred.set_title('Model Prediction', fontsize=14, fontweight='bold')
    
    # 3. Top-3 Predictions
    ax_top3 = fig.add_subplot(gs[0, 2:])
    top3_names = [CIFAR100_CLASSES[c] for c in results['top3_classes']]
    colors_top3 = ['green' if c == fine_label else 'lightblue' for c in results['top3_classes']]
    
    bars = ax_top3.barh(range(3), results['top3_probs'], color=colors_top3)
    ax_top3.set_yticks(range(3))
    ax_top3.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(top3_names)])
    ax_top3.set_xlabel('Probability')
    ax_top3.set_title('Top-3 Predictions', fontsize=14, fontweight='bold')
    ax_top3.set_xlim(0, 1)
    
    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, results['top3_probs'])):
        ax_top3.text(prob + 0.02, i, f'{prob:.2%}', va='center')
    
    # 4. Gating Network (Superclass Probabilities)
    ax_gate = fig.add_subplot(gs[1, :])
    gating_probs = results['gating_probs'].numpy()
    
    # Highlight true and predicted superclass
    colors_gate = ['lightgray'] * 20
    colors_gate[coarse_label] = 'lightgreen'  # True superclass
    if pred_coarse != coarse_label:
        colors_gate[pred_coarse] = 'lightcoral'  # Predicted superclass
    
    bars = ax_gate.bar(range(20), gating_probs, color=colors_gate, edgecolor='black', linewidth=0.5)
    ax_gate.set_xlabel('Superclass ID')
    ax_gate.set_ylabel('Probability')
    ax_gate.set_title('Gating Network: Superclass Predictions', fontsize=14, fontweight='bold')
    ax_gate.set_xticks(range(20))
    
    # Add superclass names (rotated)
    ax_gate.set_xticklabels([s[:15] + '...' if len(s) > 15 else s 
                             for s in CIFAR100_SUPERCLASSES], 
                            rotation=45, ha='right', fontsize=8)
    
    # Highlight ground truth and prediction
    bars[coarse_label].set_edgecolor('green')
    bars[coarse_label].set_linewidth(3)
    if pred_coarse != coarse_label:
        bars[pred_coarse].set_edgecolor('red')
        bars[pred_coarse].set_linewidth(3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='green', linewidth=3, label='True Superclass'),
    ]
    if pred_coarse != coarse_label:
        legend_elements.append(
            Patch(facecolor='lightcoral', edgecolor='red', linewidth=3, label='Predicted Superclass')
        )
    ax_gate.legend(handles=legend_elements, loc='upper right')
    
    # 5. Expert Network Outputs
    ax_experts = fig.add_subplot(gs[2, :])
    
    # Get the outputs from the true and predicted experts
    expert_outputs = results['expert_outputs'].numpy()  # [20, 5]
    
    # Show heatmap of all expert outputs
    im = ax_experts.imshow(expert_outputs.T, cmap='RdYlGn', aspect='auto')
    ax_experts.set_xlabel('Superclass (Expert ID)')
    ax_experts.set_ylabel('Subclass within Superclass')
    ax_experts.set_title('Expert Network Outputs (all 20 experts)', fontsize=14, fontweight='bold')
    ax_experts.set_xticks(range(20))
    ax_experts.set_yticks(range(5))
    
    # Highlight true superclass column
    ax_experts.add_patch(patches.Rectangle((coarse_label - 0.5, -0.5), 1, 5,
                                           fill=False, edgecolor='green', linewidth=3))
    
    # Highlight predicted superclass column (if different)
    if pred_coarse != coarse_label:
        ax_experts.add_patch(patches.Rectangle((pred_coarse - 0.5, -0.5), 1, 5,
                                               fill=False, edgecolor='red', linewidth=3))
    
    plt.colorbar(im, ax=ax_experts, label='Logit Value')
    
    # Add title with summary
    fig.suptitle(f'TGN Interpretability: Image Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def compare_correct_vs_incorrect(visualizer, dataloader, n_examples=5):
    """Compare correct and incorrect predictions side by side"""
    
    correct_examples = []
    incorrect_examples = []
    
    with torch.no_grad():
        for images, fine_labels, coarse_labels in dataloader:
            images = images.to(visualizer.device)
            
            for i in range(len(images)):
                if len(correct_examples) >= n_examples and len(incorrect_examples) >= n_examples:
                    break
                
                result = visualizer.analyze_image(images[i], fine_labels[i].item(), coarse_labels[i].item())
                
                if result['prediction'] == fine_labels[i].item() and len(correct_examples) < n_examples:
                    correct_examples.append((images[i].cpu(), fine_labels[i].item(), 
                                            coarse_labels[i].item(), result))
                elif result['prediction'] != fine_labels[i].item() and len(incorrect_examples) < n_examples:
                    incorrect_examples.append((images[i].cpu(), fine_labels[i].item(), 
                                              coarse_labels[i].item(), result))
            
            if len(correct_examples) >= n_examples and len(incorrect_examples) >= n_examples:
                break
    
    # Visualize comparison
    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 8))
    fig.suptitle('TGN Predictions: Correct vs Incorrect', fontsize=16, fontweight='bold')
    
    for i in range(n_examples):
        # Correct predictions (top row)
        if i < len(correct_examples):
            img, fine_label, coarse_label, result = correct_examples[i]
            ax = axes[0, i]
            ax.imshow(denormalize_image(img))
            ax.axis('off')
            
            pred_name = CIFAR100_CLASSES[result['prediction']]
            ax.set_title(f"✓ {pred_name}\n{result['confidence']:.1%}", 
                        color='green', fontweight='bold')
        
        # Incorrect predictions (bottom row)
        if i < len(incorrect_examples):
            img, fine_label, coarse_label, result = incorrect_examples[i]
            ax = axes[1, i]
            ax.imshow(denormalize_image(img))
            ax.axis('off')
            
            true_name = CIFAR100_CLASSES[fine_label]
            pred_name = CIFAR100_CLASSES[result['prediction']]
            ax.set_title(f"✗ Pred: {pred_name}\nTrue: {true_name}\n{result['confidence']:.1%}", 
                        color='red', fontweight='bold', fontsize=9)
    
    axes[0, 0].text(-0.1, 0.5, 'CORRECT', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, 
                   va='center', ha='right', color='green')
    axes[1, 0].text(-0.1, 0.5, 'INCORRECT', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, 
                   va='center', ha='right', color='red')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='TGN Interpretability Visualizer')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained TGN checkpoint')
    parser.add_argument('--image-idx', type=int, default=None,
                       help='Specific image index to visualize (default: random)')
    parser.add_argument('--compare-errors', action='store_true',
                       help='Show comparison of correct vs incorrect predictions')
    parser.add_argument('--show-features', action='store_true',
                       help='Show ResNet feature maps')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading TGN model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TaxonomicGatedNetwork().to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val accuracy: {checkpoint.get('val_acc', 0)*100:.2f}%")
    
    # Load data
    print("\nLoading CIFAR-100...")
    _, test_loader = make_dataloaders('./cifar100', batch_size=32, augment=False)
    
    # Create visualizer
    visualizer = TGNVisualizer(model, device=device)
    
    if args.compare_errors:
        print("\nGenerating error comparison...")
        fig = compare_correct_vs_incorrect(visualizer, test_loader, n_examples=5)
        save_path = output_dir / 'tgn_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
        plt.show()
    else:
        # Get a test image
        if args.image_idx is not None:
            # Get specific image
            for images, fine_labels, coarse_labels in test_loader:
                if args.image_idx < len(images):
                    image = images[args.image_idx]
                    fine_label = fine_labels[args.image_idx].item()
                    coarse_label = coarse_labels[args.image_idx].item()
                    break
                args.image_idx -= len(images)
        else:
            # Get random image
            images, fine_labels, coarse_labels = next(iter(test_loader))
            idx = torch.randint(0, len(images), (1,)).item()
            image = images[idx]
            fine_label = fine_labels[idx].item()
            coarse_label = coarse_labels[idx].item()
        
        print(f"\nAnalyzing image...")
        print(f"True class: {CIFAR100_CLASSES[fine_label]}")
        print(f"True superclass: {CIFAR100_SUPERCLASSES[coarse_label]}")
        
        # Generate main visualization
        save_path = output_dir / f'tgn_analysis_{fine_label}.png'
        fig = visualize_single_image(visualizer, image, fine_label, coarse_label, save_path)
        plt.show()
        
        # Optionally show feature maps
        if args.show_features:
            print("\nGenerating feature map visualizations...")
            results = visualizer.analyze_image(image, fine_label, coarse_label)
            
            for layer_name, features in results['layer_features'].items():
                if features is not None:
                    fig = visualize_feature_maps(features, layer_name)
                    save_path = output_dir / f'tgn_features_{layer_name}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved {layer_name} features to {save_path}")
            
            plt.show()
    
    visualizer.remove_hooks()
    print("\nDone!")


if __name__ == '__main__':
    main()
