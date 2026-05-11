from random import shuffle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms
import torch.nn as nn
from pathlib import Path
import optuna
from optuna.trial import TrialState
import argparse
import importlib
from adversarial_training_box.pipeline.early_stopper import EarlyStopper
from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule

def get_network_class(network_name):
    """Dynamically import and return network class based on name."""
    model_map = {
        'resnet18': ('resnet', 'resnet18'),
        'resnet34': ('resnet', 'resnet34'),
        'resnet50': ('resnet', 'resnet50'),
        'resnet101': ('resnet', 'resnet101'),
        'resnet152': ('resnet', 'resnet152'),
        'densenet121': ('densenet', 'densenet121'),
        'densenet169': ('densenet', 'densenet169'),
        'densenet201': ('densenet', 'densenet201'),
        'wideresnet_28_10': ('wideresnet', 'wideresnet_28_10'),
        'wideresnet_34_10': ('wideresnet', 'wideresnet_34_10'),
    }
    
    network_lower = network_name.lower()
    
    if network_lower in model_map:
        module_name, function_name = model_map[network_lower]
        try:
            module = importlib.import_module(f"adversarial_training_box.models.CIFAR.{module_name}")
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Network {network_name} not found: {e}")
        
    
if __name__ == "__main__":
    # Experiment configuration
    parser = argparse.ArgumentParser(description='Training script with configurable network and experiment name')
    parser.add_argument('--network', type=str, default="None",
                       help='Network architecture name')
    parser.add_argument('--experiment_name', type=str, default="DefaultExperiment",
                       help='Custom experiment name (default: {network}-standard-training)')
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()

    training_parameters = AttributeDict(
        learning_rate = 0.1,
        weight_decay = 5e-4,
        momentum = 0.9,
        scheduler_milestones=[60, 120, 160],
        scheduler_gamma=0.2,
        attack_epsilon=8/255,
        patience_epochs=6,
        overhead_delta=0.0,
        batch_size=128)
    
    if args.dataset == 'cifar10':
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2470, 0.2435, 0.2616]
        
        normalize = transforms.Normalize(mean=cifar_mean,std=cifar_std)

        mean_std = sum(cifar_std) / len(cifar_std)

        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 10
        full_train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        full_validation_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=test_transform)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        
        # Split with same indices for both
        train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
        _, validation_dataset = torch.utils.data.random_split(full_validation_dataset, [train_size, val_size], generator=generator)
    
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    elif args.dataset == 'cifar100':
        cifar_mean = [0.5071, 0.4865, 0.4409]
        cifar_std = [0.2673, 0.2564, 0.2762]
        
        normalize = transforms.Normalize(mean=cifar_mean,std=cifar_std)

        mean_std = sum(cifar_std) / len(cifar_std)

        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 100
        full_train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
        full_validation_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=test_transform)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        generator = torch.Generator().manual_seed(43)
        
        # Split with same indices for both
        train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
        _, validation_dataset = torch.utils.data.random_split(full_validation_dataset, [train_size, val_size], generator=generator)
    
        test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)


    experiment_name = args.experiment_name
    network_function = get_network_class(args.network)
    network = network_function(num_classes=num_classes)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)    

    # Training configuration
    optimizer = getattr(optim, 'SGD')(network.parameters(), lr=training_parameters.learning_rate, weight_decay=training_parameters.weight_decay, momentum=training_parameters.momentum)
    scheduler = MultiStepLR(optimizer, milestones=training_parameters.scheduler_milestones, gamma=training_parameters.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=training_parameters.patience_epochs, delta=training_parameters.overhead_delta)

    # Validation module
    validation_module = StandardTestModule(criterion=criterion)

    # Training modules stack
    training_stack = []
    training_stack.append((200, StandardTrainingModule(criterion=criterion)))

    # Testing modules stack
    testing_stack = [StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=2/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=4/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=8/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=2/255/mean_std/4, number_iterations=20, random_init=True), epsilon=2/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=4/255/mean_std/4, number_iterations=20, random_init=True), epsilon=4/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=8/255/mean_std/4, number_iterations=20, random_init=True), epsilon=8/255/mean_std),
    ]
    
    # Convert complex objects to JSON-serializable format
    def serialize_training_stack(stack):
        return [{"epochs": epochs, "module_type": type(module).__name__, 
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for epochs, module in stack]
    
    def serialize_testing_stack(stack):
        return [{"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for module in stack]
    
    def serialize_validation_module(module):
        return {"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)}

    training_objects = AttributeDict(criterion=str(criterion), 
                                     optimizer=str(optimizer), 
                                     network=str(network), 
                                     scheduler=str(scheduler), 
                                     training_stack=serialize_training_stack(training_stack),
                                     testing_stack=serialize_testing_stack(testing_stack),
                                     validation_module=serialize_validation_module(validation_module))

    # Setup experiment
    experiment_tracker = ExperimentTracker(experiment_name, Path("./generated"), login=True)
    experiment_tracker.initialize_new_experiment("", training_parameters=training_parameters | training_objects)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)

    # Train
    pipeline.train(train_loader, network, training_stack, early_stopper=None, 
                   validation_loader=validation_loader,
                   validation_module=validation_module
                   )

    # Test
    network = experiment_tracker.load_trained_model(network)
    pipeline.test(network, test_loader, testing_stack=testing_stack)
    experiment_tracker.export_to_onnx(network, test_loader)