import argparse
from config import Config
from trainer import train


def main():
    parser = argparse.ArgumentParser(description="Mapping Networks Training")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--cnn_spec", type=str, default="large",
                        choices=["default", "large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--latent_init_std", type=float, default=1.0)
    parser.add_argument("--output_gain", type=float, default=1.0)
    parser.add_argument("--lambda_stability", type=float, default=0.01)
    parser.add_argument("--lambda_smoothness", type=float, default=0.1)
    parser.add_argument("--lambda_alignment", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="mapping_mnist")
    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        cnn_spec=args.cnn_spec,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        latent_init_std=args.latent_init_std,
        output_gain=args.output_gain,
        lambda_stability=args.lambda_stability,
        lambda_smoothness=args.lambda_smoothness,
        lambda_alignment=args.lambda_alignment,
        seed=args.seed,
        device=args.device,
        exp_name=args.exp_name,
    )

    print("=" * 60)
    print("Mapping Networks Training")
    print("=" * 60)
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    mapping, best_acc = train(cfg)
    return mapping, best_acc


if __name__ == "__main__":
    main()
