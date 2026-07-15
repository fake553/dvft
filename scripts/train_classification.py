import argparse
import torch
from dvft.models import DVFTAnchor, DVFTQuery
from dvft.models.fusion_baselines import (
    SingleBackboneAdapter, TransformerFusion, 
    ConcatFusion, AddFusion, GatingFusion, MLPFusion
)
from dvft.utils.data import get_imagenet_loader, get_cifar100_loader
from dvft.utils.train import train_classification, evaluate
from dvft.utils.helpers import set_seed


def get_model(args):
    if args.model == 'dvft_anchor':
        return DVFTAnchor(num_classes=args.num_classes)
    elif args.model == 'dvft_query':
        return DVFTQuery(num_classes=args.num_classes)
    elif args.model == 'single_adapter':
        return SingleBackboneAdapter(num_classes=args.num_classes)
    elif args.model == 'transformer_fusion':
        return TransformerFusion(768, 1024, num_classes=args.num_classes)
    elif args.model == 'concat':
        return ConcatFusion(768, 1024, num_classes=args.num_classes)
    elif args.model == 'add':
        return AddFusion(768, 1024, num_classes=args.num_classes)
    elif args.model == 'gating':
        return GatingFusion(768, 1024, num_classes=args.num_classes)
    elif args.model == 'mlp':
        return MLPFusion(768, 1024, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dvft_anchor',
                        choices=['dvft_anchor', 'dvft_query', 'single_adapter',
                                 'transformer_fusion', 'concat', 'add', 'gating', 'mlp'])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar100'])
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    if args.dataset == 'imagenet':
        train_loader, val_loader, num_classes = get_imagenet_loader(args.data_dir, args.batch_size)
    else:
        train_loader, val_loader, num_classes = get_cifar100_loader(args.data_dir, args.batch_size)
    
    args.num_classes = num_classes
    model = get_model(args)
    model.to(device)
    
    train_classification(model, train_loader, val_loader, device,
                         epochs=args.epochs, num_classes=num_classes,
                         save_path=args.save_path)


if __name__ == '__main__':
    main()