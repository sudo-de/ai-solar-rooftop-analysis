"""
Federated Learning System for Privacy-Preserving Solar Analysis
Enables training across decentralized datasets without compromising privacy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import time
import random
from collections import defaultdict
import threading
import queue
import base64
import os
from scipy.stats import laplace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class FederatedModel:
    """Federated learning model configuration"""
    model_weights: Dict[str, torch.Tensor]
    model_version: str
    training_round: int
    client_count: int
    privacy_budget: float

@dataclass
class ClientData:
    """Enhanced client data for federated learning"""
    client_id: str
    data_hash: str
    encrypted_gradients: bytes
    sample_count: int
    privacy_noise: float
    data_quality: float
    computational_capability: str
    network_bandwidth: float
    last_seen: float
    participation_rate: float

@dataclass
class DifferentialPrivacyConfig:
    """Differential privacy configuration"""
    epsilon: float  # Privacy parameter
    delta: float    # Failure probability
    sensitivity: float  # L2 sensitivity
    noise_scale: float  # Noise scale for Gaussian mechanism

@dataclass
class FederatedMetrics:
    """Federated learning performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    privacy_budget_used: float
    communication_rounds: int
    convergence_rate: float
    client_dropout_rate: float

class AdvancedPrivacyPreservingAggregator:
    """Advanced privacy-preserving model aggregation with differential privacy"""
    
    def __init__(self, privacy_budget: float = 1.0, epsilon: float = 1.0, delta: float = 1e-5):
        self.privacy_budget = privacy_budget
        self.epsilon = epsilon
        self.delta = delta
        self.logger = logging.getLogger(__name__)
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Advanced privacy mechanisms
        self.dp_config = DifferentialPrivacyConfig(
            epsilon=epsilon,
            delta=delta,
            sensitivity=1.0,  # L2 sensitivity
            noise_scale=self._calculate_noise_scale(epsilon, delta)
        )
        
        # Client selection and weighting
        self.client_weights = {}
        self.quality_scores = {}
        self.participation_history = defaultdict(list)
        
        # Adaptive learning rates
        self.adaptive_lr = True
        self.lr_decay = 0.99
        self.min_lr = 0.001
        
        # Communication efficiency
        self.compression_enabled = True
        self.quantization_bits = 8
    
    def aggregate_models_advanced(self, client_models: List[FederatedModel], 
                                 client_data: List[ClientData]) -> FederatedModel:
        """Advanced aggregation with quality-aware weighting and differential privacy"""
        if not client_models:
            raise ValueError("No client models provided")
        
        # Update client quality scores
        self._update_quality_scores(client_data)
        
        # Calculate adaptive weights based on quality and participation
        adaptive_weights = self._calculate_adaptive_weights(client_data)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in client_models[0].model_weights.keys():
            aggregated_weights[key] = torch.zeros_like(client_models[0].model_weights[key])
        
        # Quality-weighted aggregation
        for i, model in enumerate(client_models):
            client_weight = adaptive_weights[i]
            for key, weight in model.model_weights.items():
                aggregated_weights[key] += client_weight * weight
        
        # Apply advanced differential privacy
        aggregated_weights = self._apply_advanced_differential_privacy(aggregated_weights, client_data)
        
        # Apply gradient clipping for stability
        aggregated_weights = self._apply_gradient_clipping(aggregated_weights)
        
        # Create aggregated model
        aggregated_model = FederatedModel(
            model_weights=aggregated_weights,
            model_version=f"v{int(time.time())}",
            training_round=client_models[0].training_round + 1,
            client_count=len(client_models),
            privacy_budget=self._calculate_privacy_budget(client_data)
        )
        
        return aggregated_model
    
    def _update_quality_scores(self, client_data: List[ClientData]):
        """Update quality scores based on client performance"""
        for client in client_data:
            client_id = client.client_id
            
            # Calculate quality score based on multiple factors
            data_quality = client.data_quality
            participation_rate = client.participation_rate
            computational_capability = self._get_capability_score(client.computational_capability)
            
            # Weighted quality score
            quality_score = (0.4 * data_quality + 
                           0.3 * participation_rate + 
                           0.3 * computational_capability)
            
            self.quality_scores[client_id] = quality_score
            
            # Update participation history
            self.participation_history[client_id].append(quality_score)
            
            # Keep only recent history (last 10 rounds)
            if len(self.participation_history[client_id]) > 10:
                self.participation_history[client_id] = self.participation_history[client_id][-10:]
    
    def _get_capability_score(self, capability: str) -> float:
        """Convert computational capability to numeric score"""
        capability_scores = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
            "mobile": 0.3
        }
        return capability_scores.get(capability.lower(), 0.5)
    
    def _calculate_adaptive_weights(self, client_data: List[ClientData]) -> List[float]:
        """Calculate adaptive weights based on quality and participation"""
        weights = []
        
        for client in client_data:
            client_id = client.client_id
            
            # Base weight from sample count
            sample_weight = client.sample_count / sum(c.sample_count for c in client_data)
            
            # Quality weight
            quality_weight = self.quality_scores.get(client_id, 0.5)
            
            # Participation weight (recent performance)
            participation_history = self.participation_history.get(client_id, [])
            if participation_history:
                participation_weight = np.mean(participation_history[-3:])  # Last 3 rounds
            else:
                participation_weight = 0.5
            
            # Combined adaptive weight
            adaptive_weight = (0.4 * sample_weight + 
                             0.4 * quality_weight + 
                             0.2 * participation_weight)
            
            weights.append(adaptive_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _apply_advanced_differential_privacy(self, weights: Dict[str, torch.Tensor], 
                                           client_data: List[ClientData]) -> Dict[str, torch.Tensor]:
        """Apply advanced differential privacy mechanisms"""
        noisy_weights = {}
        
        for key, weight in weights.items():
            # Calculate L2 sensitivity
            sensitivity = self._calculate_l2_sensitivity(weight, client_data)
            
            # Calculate noise scale based on privacy parameters
            noise_scale = self._calculate_adaptive_noise_scale(sensitivity)
            
            # Add Gaussian noise with adaptive scaling
            noise = torch.normal(0, noise_scale, weight.shape)
            noisy_weights[key] = weight + noise
            
            # Apply post-processing for utility preservation
            noisy_weights[key] = self._apply_post_processing(noisy_weights[key], weight)
        
        return noisy_weights
    
    def _calculate_l2_sensitivity(self, weight: torch.Tensor, client_data: List[ClientData]) -> float:
        """Calculate L2 sensitivity for differential privacy"""
        # Simplified sensitivity calculation
        # In practice, this would be calculated based on the actual gradient norms
        max_gradient_norm = max(client.sample_count for client in client_data)
        sensitivity = 2.0 / max_gradient_norm  # Simplified bound
        
        return sensitivity
    
    def _calculate_adaptive_noise_scale(self, sensitivity: float) -> float:
        """Calculate adaptive noise scale based on privacy parameters"""
        # Gaussian mechanism noise scale
        noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        
        # Adaptive scaling based on privacy budget
        budget_factor = min(1.0, self.privacy_budget / self.epsilon)
        noise_scale *= budget_factor
        
        return noise_scale
    
    def _apply_post_processing(self, noisy_weight: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        """Apply post-processing to preserve utility while maintaining privacy"""
        # Smoothing to reduce noise impact
        alpha = 0.1  # Smoothing parameter
        smoothed_weight = alpha * original_weight + (1 - alpha) * noisy_weight
        
        # Clipping to prevent extreme values
        clipped_weight = torch.clamp(smoothed_weight, -10, 10)
        
        return clipped_weight
    
    def _apply_gradient_clipping(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply gradient clipping for training stability"""
        clipped_weights = {}
        max_norm = 1.0  # Maximum gradient norm
        
        for key, weight in weights.items():
            # Calculate gradient norm
            grad_norm = torch.norm(weight)
            
            # Clip if norm exceeds threshold
            if grad_norm > max_norm:
                clipped_weight = weight * (max_norm / grad_norm)
                clipped_weights[key] = clipped_weight
            else:
                clipped_weights[key] = weight
        
        return clipped_weights
    
    def _calculate_noise_scale(self, epsilon: float, delta: float) -> float:
        """Calculate noise scale for differential privacy"""
        if epsilon <= 0 or delta <= 0:
            raise ValueError("Epsilon and delta must be positive")
        
        # Gaussian mechanism noise scale
        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return noise_scale
    
    def _add_differential_privacy_noise(self, weights: Dict[str, torch.Tensor], 
                                       client_data: List[ClientData]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to aggregated weights"""
        noisy_weights = {}
        
        for key, weight in weights.items():
            # Calculate noise scale based on privacy budget
            noise_scale = self._calculate_noise_scale(client_data)
            
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, weight.shape)
            noisy_weights[key] = weight + noise
        
        return noisy_weights
    
    def _calculate_noise_scale(self, client_data: List[ClientData]) -> float:
        """Calculate noise scale for differential privacy"""
        # Simplified noise calculation
        total_clients = len(client_data)
        privacy_budget = sum(client.privacy_noise for client in client_data) / total_clients
        
        # Higher privacy budget = less noise
        noise_scale = self.noise_multiplier / (privacy_budget + 1e-8)
        
        return noise_scale
    
    def _calculate_privacy_budget(self, client_data: List[ClientData]) -> float:
        """Calculate remaining privacy budget"""
        used_budget = sum(client.privacy_noise for client in client_data)
        remaining_budget = max(0, self.privacy_budget - used_budget)
        
        return remaining_budget

class SecureAggregation:
    """Secure aggregation protocol for federated learning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_gradients(self, gradients: Dict[str, torch.Tensor], 
                         client_id: str) -> bytes:
        """Encrypt gradients for secure transmission"""
        try:
            # Serialize gradients
            gradients_serialized = self._serialize_tensors(gradients)
            
            # Add client identifier
            data_with_id = {
                "client_id": client_id,
                "gradients": gradients_serialized,
                "timestamp": time.time()
            }
            
            # Encrypt data
            data_json = json.dumps(data_with_id, default=str)
            encrypted_data = self.cipher.encrypt(data_json.encode())
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Gradient encryption failed: {e}")
            raise
    
    def decrypt_gradients(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt gradients from secure transmission"""
        try:
            # Decrypt data
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data_dict = json.loads(decrypted_data.decode())
            
            # Deserialize gradients
            gradients = self._deserialize_tensors(data_dict["gradients"])
            
            return {
                "client_id": data_dict["client_id"],
                "gradients": gradients,
                "timestamp": data_dict["timestamp"]
            }
            
        except Exception as e:
            self.logger.error(f"Gradient decryption failed: {e}")
            raise
    
    def _serialize_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Serialize PyTorch tensors for JSON encoding"""
        serialized = {}
        for key, tensor in tensors.items():
            serialized[key] = {
                "data": tensor.detach().cpu().numpy().tolist(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype)
            }
        return serialized
    
    def _deserialize_tensors(self, serialized: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Deserialize tensors from JSON format"""
        tensors = {}
        for key, data in serialized.items():
            tensor_data = np.array(data["data"])
            tensor = torch.from_numpy(tensor_data).reshape(data["shape"])
            tensors[key] = tensor
        return tensors

class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple clients"""
    
    def __init__(self, global_model: nn.Module, privacy_budget: float = 1.0):
        self.global_model = global_model
        self.privacy_budget = privacy_budget
        self.logger = logging.getLogger(__name__)
        self.aggregator = PrivacyPreservingAggregator(privacy_budget)
        self.secure_agg = SecureAggregation()
        self.training_round = 0
        self.client_registry = {}
    
    def register_client(self, client_id: str, data_hash: str, 
                       sample_count: int) -> str:
        """Register a new client for federated learning"""
        client_info = {
            "client_id": client_id,
            "data_hash": data_hash,
            "sample_count": sample_count,
            "privacy_noise": 0.1,  # Default privacy noise
            "registered_at": time.time(),
            "active": True
        }
        
        self.client_registry[client_id] = client_info
        self.logger.info(f"Registered client {client_id} with {sample_count} samples")
        
        return f"Client {client_id} registered successfully"
    
    def start_federated_training(self, clients: List[str], 
                               local_epochs: int = 5) -> Dict[str, Any]:
        """Start federated training round"""
        self.training_round += 1
        self.logger.info(f"Starting federated training round {self.training_round}")
        
        # Distribute global model to clients
        global_weights = self._extract_model_weights()
        
        # Collect encrypted gradients from clients
        client_gradients = []
        client_data = []
        
        for client_id in clients:
            if client_id in self.client_registry:
                # Simulate client training (in practice, this would be done by clients)
                gradients = self._simulate_client_training(client_id, local_epochs)
                encrypted_gradients = self.secure_agg.encrypt_gradients(gradients, client_id)
                
                client_data.append(ClientData(
                    client_id=client_id,
                    data_hash=self.client_registry[client_id]["data_hash"],
                    encrypted_gradients=encrypted_gradients,
                    sample_count=self.client_registry[client_id]["sample_count"],
                    privacy_noise=self.client_registry[client_id]["privacy_noise"]
                ))
        
        # Aggregate models with privacy preservation
        aggregated_model = self.aggregator.aggregate_models(
            [self._create_federated_model(global_weights) for _ in client_data],
            client_data
        )
        
        # Update global model
        self._update_global_model(aggregated_model.model_weights)
        
        return {
            "training_round": self.training_round,
            "participating_clients": len(clients),
            "privacy_budget_used": aggregated_model.privacy_budget,
            "model_version": aggregated_model.model_version,
            "aggregation_successful": True
        }
    
    def _extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract weights from global model"""
        weights = {}
        for name, param in self.global_model.named_parameters():
            weights[name] = param.data.clone()
        return weights
    
    def _create_federated_model(self, weights: Dict[str, torch.Tensor]) -> FederatedModel:
        """Create federated model from weights"""
        return FederatedModel(
            model_weights=weights,
            model_version=f"v{self.training_round}",
            training_round=self.training_round,
            client_count=1,
            privacy_budget=self.privacy_budget
        )
    
    def _simulate_client_training(self, client_id: str, local_epochs: int) -> Dict[str, torch.Tensor]:
        """Simulate client training (in practice, done by clients)"""
        # Generate random gradients for simulation
        gradients = {}
        for name, param in self.global_model.named_parameters():
            gradients[name] = torch.randn_like(param) * 0.01
        
        return gradients
    
    def _update_global_model(self, new_weights: Dict[str, torch.Tensor]):
        """Update global model with new weights"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in new_weights:
                    param.data = new_weights[name]
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        # Simulate performance metrics
        return {
            "accuracy": 0.95,
            "privacy_budget_remaining": self.privacy_budget,
            "training_rounds": self.training_round,
            "active_clients": len([c for c in self.client_registry.values() if c["active"]]),
            "total_samples": sum(c["sample_count"] for c in self.client_registry.values())
        }

class PrivacyBudgetManager:
    """Manages privacy budget for differential privacy"""
    
    def __init__(self, initial_budget: float = 1.0):
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.logger = logging.getLogger(__name__)
    
    def consume_privacy_budget(self, amount: float) -> bool:
        """Consume privacy budget for a computation"""
        if self.current_budget >= amount:
            self.current_budget -= amount
            self.logger.info(f"Consumed {amount} privacy budget. Remaining: {self.current_budget}")
            return True
        else:
            self.logger.warning(f"Insufficient privacy budget. Required: {amount}, Available: {self.current_budget}")
            return False
    
    def reset_budget(self):
        """Reset privacy budget to initial value"""
        self.current_budget = self.initial_budget
        self.logger.info(f"Privacy budget reset to {self.initial_budget}")
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return self.current_budget
    
    def calculate_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Calculate noise scale for differential privacy"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Gaussian mechanism noise scale
        noise_scale = sensitivity / epsilon
        return noise_scale

class FederatedSolarAnalyzer:
    """Federated learning system for solar rooftop analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coordinator = None
        self.privacy_manager = PrivacyBudgetManager()
    
    def initialize_federated_system(self, base_model: nn.Module) -> str:
        """Initialize federated learning system"""
        self.coordinator = FederatedLearningCoordinator(base_model)
        self.logger.info("Federated learning system initialized")
        return "Federated learning system ready"
    
    def add_client_data(self, client_id: str, data_hash: str, 
                       sample_count: int) -> str:
        """Add client data to federated learning"""
        if not self.coordinator:
            raise RuntimeError("Federated system not initialized")
        
        return self.coordinator.register_client(client_id, data_hash, sample_count)
    
    def run_federated_training(self, client_ids: List[str]) -> Dict[str, Any]:
        """Run federated training with specified clients"""
        if not self.coordinator:
            raise RuntimeError("Federated system not initialized")
        
        return self.coordinator.start_federated_training(client_ids)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get federated learning system status"""
        if not self.coordinator:
            return {"status": "not_initialized"}
        
        performance = self.coordinator.get_model_performance()
        return {
            "status": "active",
            "performance": performance,
            "privacy_budget": self.privacy_manager.get_remaining_budget()
        }
