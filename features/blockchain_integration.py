"""
Blockchain Integration for Solar Potential Data Verification
Enables trust, transparency, and incentive distribution
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

@dataclass
class SolarDataBlock:
    """Blockchain block for solar data"""
    block_id: str
    previous_hash: str
    timestamp: float
    solar_data: Dict[str, Any]
    data_hash: str
    validator_signature: str
    merkle_root: str

@dataclass
class IncentiveToken:
    """Incentive token for solar data contribution"""
    token_id: str
    recipient_address: str
    amount: float
    token_type: str  # "carbon_credit", "renewable_energy_credit", "data_contribution"
    timestamp: float
    transaction_hash: str

@dataclass
class SmartContract:
    """Smart contract for solar data verification"""
    contract_address: str
    contract_type: str
    conditions: Dict[str, Any]
    rewards: Dict[str, float]
    active: bool

class SolarDataBlockchain:
    """Blockchain system for solar potential data verification"""
    
    def __init__(self, network: str = "ethereum_testnet"):
        self.network = network
        self.logger = logging.getLogger(__name__)
        self.blocks = []
        self.pending_transactions = []
        self.validators = []
        self.smart_contracts = {}
    
    def create_solar_data_block(self, solar_data: Dict[str, Any], 
                              validator_id: str) -> SolarDataBlock:
        """Create a new block for solar data"""
        try:
            # Calculate data hash
            data_hash = self._calculate_data_hash(solar_data)
            
            # Get previous block hash
            previous_hash = self.blocks[-1].data_hash if self.blocks else "0"
            
            # Create block
            block = SolarDataBlock(
                block_id=f"block_{len(self.blocks) + 1}",
                previous_hash=previous_hash,
                timestamp=time.time(),
                solar_data=solar_data,
                data_hash=data_hash,
                validator_signature=self._sign_data(solar_data, validator_id),
                merkle_root=self._calculate_merkle_root(solar_data)
            )
            
            # Add to blockchain
            self.blocks.append(block)
            
            self.logger.info(f"Created block {block.block_id} with {len(solar_data)} data points")
            return block
            
        except Exception as e:
            self.logger.error(f"Block creation failed: {e}")
            raise
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of solar data"""
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _sign_data(self, data: Dict[str, Any], validator_id: str) -> str:
        """Sign data with validator signature"""
        # Simplified signature - in practice, use proper cryptographic signing
        signature_data = f"{validator_id}_{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(signature_data.encode()).hexdigest()
    
    def _calculate_merkle_root(self, data: Dict[str, Any]) -> str:
        """Calculate Merkle root for data integrity"""
        # Simplified Merkle root calculation
        data_items = [f"{k}:{v}" for k, v in data.items()]
        
        if len(data_items) == 1:
            return hashlib.sha256(data_items[0].encode()).hexdigest()
        
        # Build Merkle tree
        current_level = [hashlib.sha256(item.encode()).hexdigest() for item in data_items]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            current_level = next_level
        
        return current_level[0]
    
    def verify_data_integrity(self, block_id: str) -> bool:
        """Verify data integrity of a specific block"""
        try:
            block = next((b for b in self.blocks if b.block_id == block_id), None)
            if not block:
                return False
            
            # Verify hash chain
            if block.previous_hash != (self.blocks[self.blocks.index(block) - 1].data_hash if self.blocks.index(block) > 0 else "0"):
                return False
            
            # Verify data hash
            calculated_hash = self._calculate_data_hash(block.solar_data)
            if calculated_hash != block.data_hash:
                return False
            
            # Verify Merkle root
            calculated_merkle = self._calculate_merkle_root(block.solar_data)
            if calculated_merkle != block.merkle_root:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data integrity verification failed: {e}")
            return False
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain status"""
        return {
            "total_blocks": len(self.blocks),
            "network": self.network,
            "validators": len(self.validators),
            "pending_transactions": len(self.pending_transactions),
            "last_block_time": self.blocks[-1].timestamp if self.blocks else None,
            "blockchain_integrity": self._verify_blockchain_integrity()
        }
    
    def _verify_blockchain_integrity(self) -> bool:
        """Verify integrity of entire blockchain"""
        for i, block in enumerate(self.blocks):
            if not self.verify_data_integrity(block.block_id):
                return False
        return True

class IncentiveDistributionSystem:
    """System for distributing incentives based on solar data contributions"""
    
    def __init__(self, blockchain: SolarDataBlockchain):
        self.blockchain = blockchain
        self.logger = logging.getLogger(__name__)
        self.incentive_tokens = []
        self.reward_pools = {
            "carbon_credits": 10000.0,
            "renewable_energy_credits": 5000.0,
            "data_contribution_tokens": 2000.0
        }
    
    def distribute_incentives(self, contributor_address: str, 
                            solar_data: Dict[str, Any],
                            data_quality_score: float) -> List[IncentiveToken]:
        """Distribute incentives based on data contribution"""
        try:
            tokens = []
            
            # Calculate base reward
            base_reward = self._calculate_base_reward(solar_data, data_quality_score)
            
            # Carbon credit reward
            if solar_data.get("annual_energy_kwh", 0) > 1000:
                carbon_credits = base_reward * 0.1  # 10% of base reward
                if carbon_credits > 0:
                    carbon_token = IncentiveToken(
                        token_id=f"carbon_{int(time.time())}",
                        recipient_address=contributor_address,
                        amount=carbon_credits,
                        token_type="carbon_credit",
                        timestamp=time.time(),
                        transaction_hash=self._generate_transaction_hash()
                    )
                    tokens.append(carbon_token)
                    self.reward_pools["carbon_credits"] -= carbon_credits
            
            # Renewable energy credit
            if solar_data.get("system_size_kw", 0) > 5:
                renewable_credits = base_reward * 0.05  # 5% of base reward
                if renewable_credits > 0:
                    renewable_token = IncentiveToken(
                        token_id=f"renewable_{int(time.time())}",
                        recipient_address=contributor_address,
                        amount=renewable_credits,
                        token_type="renewable_energy_credit",
                        timestamp=time.time(),
                        transaction_hash=self._generate_transaction_hash()
                    )
                    tokens.append(renewable_token)
                    self.reward_pools["renewable_energy_credits"] -= renewable_credits
            
            # Data contribution token
            data_tokens = base_reward * 0.02  # 2% of base reward
            if data_tokens > 0:
                data_token = IncentiveToken(
                    token_id=f"data_{int(time.time())}",
                    recipient_address=contributor_address,
                    amount=data_tokens,
                    token_type="data_contribution",
                    timestamp=time.time(),
                    transaction_hash=self._generate_transaction_hash()
                )
                tokens.append(data_token)
                self.reward_pools["data_contribution_tokens"] -= data_tokens
            
            # Add tokens to blockchain
            self.incentive_tokens.extend(tokens)
            
            self.logger.info(f"Distributed {len(tokens)} incentive tokens to {contributor_address}")
            return tokens
            
        except Exception as e:
            self.logger.error(f"Incentive distribution failed: {e}")
            raise
    
    def _calculate_base_reward(self, solar_data: Dict[str, Any], quality_score: float) -> float:
        """Calculate base reward amount"""
        # Base reward factors
        energy_factor = solar_data.get("annual_energy_kwh", 0) / 10000  # Normalize to 10kWh
        area_factor = solar_data.get("usable_area_m2", 0) / 100  # Normalize to 100m²
        quality_factor = quality_score / 10  # Normalize to 10
        
        # Calculate base reward
        base_reward = (energy_factor + area_factor + quality_factor) * 100
        
        return max(0, base_reward)
    
    def _generate_transaction_hash(self) -> str:
        """Generate transaction hash"""
        transaction_data = f"{time.time()}_{np.random.random()}"
        return hashlib.sha256(transaction_data.encode()).hexdigest()
    
    def get_incentive_balance(self, address: str) -> Dict[str, float]:
        """Get incentive balance for an address"""
        balance = {
            "carbon_credits": 0.0,
            "renewable_energy_credits": 0.0,
            "data_contribution_tokens": 0.0
        }
        
        for token in self.incentive_tokens:
            if token.recipient_address == address:
                if token.token_type == "carbon_credit":
                    balance["carbon_credits"] += token.amount
                elif token.token_type == "renewable_energy_credit":
                    balance["renewable_energy_credits"] += token.amount
                elif token.token_type == "data_contribution":
                    balance["data_contribution_tokens"] += token.amount
        
        return balance

class SmartContractManager:
    """Manage smart contracts for solar data verification and rewards"""
    
    def __init__(self, blockchain: SolarDataBlockchain):
        self.blockchain = blockchain
        self.logger = logging.getLogger(__name__)
        self.contracts = {}
    
    def deploy_contract(self, contract_type: str, conditions: Dict[str, Any]) -> str:
        """Deploy a new smart contract"""
        try:
            contract_address = f"contract_{len(self.contracts) + 1}_{int(time.time())}"
            
            contract = SmartContract(
                contract_address=contract_address,
                contract_type=contract_type,
                conditions=conditions,
                rewards=self._calculate_contract_rewards(contract_type),
                active=True
            )
            
            self.contracts[contract_address] = contract
            
            self.logger.info(f"Deployed {contract_type} contract at {contract_address}")
            return contract_address
            
        except Exception as e:
            self.logger.error(f"Contract deployment failed: {e}")
            raise
    
    def _calculate_contract_rewards(self, contract_type: str) -> Dict[str, float]:
        """Calculate rewards for different contract types"""
        reward_structures = {
            "data_verification": {
                "verification_reward": 10.0,
                "quality_bonus": 5.0,
                "consistency_bonus": 3.0
            },
            "carbon_offset": {
                "carbon_credit": 0.1,
                "verification_bonus": 2.0
            },
            "renewable_energy": {
                "energy_credit": 0.05,
                "efficiency_bonus": 1.0
            }
        }
        
        return reward_structures.get(contract_type, {})
    
    def execute_contract(self, contract_address: str, 
                        solar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart contract with solar data"""
        try:
            contract = self.contracts.get(contract_address)
            if not contract or not contract.active:
                raise ValueError(f"Contract {contract_address} not found or inactive")
            
            # Check contract conditions
            conditions_met = self._check_contract_conditions(contract, solar_data)
            
            if conditions_met:
                # Calculate rewards
                rewards = self._calculate_rewards(contract, solar_data)
                
                # Execute contract
                execution_result = {
                    "contract_address": contract_address,
                    "conditions_met": True,
                    "rewards": rewards,
                    "execution_time": time.time(),
                    "transaction_hash": self._generate_transaction_hash()
                }
                
                self.logger.info(f"Contract {contract_address} executed successfully")
                return execution_result
            else:
                return {
                    "contract_address": contract_address,
                    "conditions_met": False,
                    "rewards": {},
                    "execution_time": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Contract execution failed: {e}")
            raise
    
    def _check_contract_conditions(self, contract: SmartContract, 
                                  solar_data: Dict[str, Any]) -> bool:
        """Check if contract conditions are met"""
        conditions = contract.conditions
        
        # Check minimum energy requirement
        if "min_energy_kwh" in conditions:
            if solar_data.get("annual_energy_kwh", 0) < conditions["min_energy_kwh"]:
                return False
        
        # Check minimum area requirement
        if "min_area_m2" in conditions:
            if solar_data.get("usable_area_m2", 0) < conditions["min_area_m2"]:
                return False
        
        # Check quality requirements
        if "min_quality_score" in conditions:
            if solar_data.get("suitability_score", 0) < conditions["min_quality_score"]:
                return False
        
        return True
    
    def _calculate_rewards(self, contract: SmartContract, 
                          solar_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards based on contract and data"""
        rewards = {}
        
        for reward_type, base_amount in contract.rewards.items():
            if reward_type == "verification_reward":
                rewards[reward_type] = base_amount
            elif reward_type == "quality_bonus":
                quality_score = solar_data.get("suitability_score", 5)
                rewards[reward_type] = base_amount * (quality_score / 10)
            elif reward_type == "carbon_credit":
                energy_kwh = solar_data.get("annual_energy_kwh", 0)
                rewards[reward_type] = base_amount * (energy_kwh / 1000)
            elif reward_type == "energy_credit":
                system_size = solar_data.get("system_size_kw", 0)
                rewards[reward_type] = base_amount * system_size
        
        return rewards
    
    def _generate_transaction_hash(self) -> str:
        """Generate transaction hash"""
        transaction_data = f"{time.time()}_{np.random.random()}"
        return hashlib.sha256(transaction_data.encode()).hexdigest()

class SolarDataVerificationSystem:
    """Complete system for solar data verification and incentive distribution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blockchain = SolarDataBlockchain()
        self.incentive_system = IncentiveDistributionSystem(self.blockchain)
        self.contract_manager = SmartContractManager(self.blockchain)
    
    def submit_solar_data(self, solar_data: Dict[str, Any], 
                         contributor_address: str,
                         validator_id: str) -> Dict[str, Any]:
        """Submit solar data for verification and reward distribution"""
        try:
            # Create blockchain block
            block = self.blockchain.create_solar_data_block(solar_data, validator_id)
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality(solar_data)
            
            # Distribute incentives
            incentives = self.incentive_system.distribute_incentives(
                contributor_address, solar_data, quality_score
            )
            
            # Execute smart contracts
            contract_results = []
            for contract_address in self.contract_manager.contracts:
                result = self.contract_manager.execute_contract(contract_address, solar_data)
                contract_results.append(result)
            
            return {
                "block_id": block.block_id,
                "data_hash": block.data_hash,
                "quality_score": quality_score,
                "incentives": [asdict(token) for token in incentives],
                "contract_results": contract_results,
                "verification_status": "verified"
            }
            
        except Exception as e:
            self.logger.error(f"Solar data submission failed: {e}")
            raise
    
    def _calculate_data_quality(self, solar_data: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        quality_factors = []
        
        # Energy data quality
        if "annual_energy_kwh" in solar_data and solar_data["annual_energy_kwh"] > 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Area data quality
        if "usable_area_m2" in solar_data and solar_data["usable_area_m2"] > 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Suitability data quality
        if "suitability_score" in solar_data and solar_data["suitability_score"] > 0:
            quality_factors.append(solar_data["suitability_score"] / 10)
        else:
            quality_factors.append(0.0)
        
        # Calculate average quality score
        return sum(quality_factors) / len(quality_factors) * 10
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "blockchain": self.blockchain.get_blockchain_status(),
            "incentive_pools": self.incentive_system.reward_pools,
            "active_contracts": len(self.contract_manager.contracts),
            "total_incentives_distributed": len(self.incentive_system.incentive_tokens)
        }
