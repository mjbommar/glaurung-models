#!/usr/bin/env python3
"""Ultra-hard synthetic data generator combining real IOCs with adversarial examples."""

import random
import string
import hashlib
import ipaddress
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json
import logging

from .enhanced_synthetic_gen import EnhancedSyntheticGenerator, SyntheticSample
from .real_ioc_fetcher import RealIOCFetcher, AdversarialIOCGenerator
from .ioc_patterns import IOCType
from . import fallback_iocs

logger = logging.getLogger(__name__)


class UltraHardDataGenerator(EnhancedSyntheticGenerator):
    """Generate extremely challenging training data with real IOCs and adversarial examples."""
    
    def __init__(self, seed: Optional[int] = None, use_real_iocs: bool = True):
        super().__init__(seed)
        
        self.use_real_iocs = use_real_iocs
        self.real_fetcher = RealIOCFetcher() if use_real_iocs else None
        self.adversarial_gen = AdversarialIOCGenerator()
        
        # Real IOCs cache
        self.real_iocs = {
            'domains': [],
            'urls': [],
            'ips': [],
            'hashes_md5': [],
            'hashes_sha256': [],
        }
        
        # Load real IOCs if enabled
        if self.use_real_iocs:
            self._load_real_iocs()
        
        # Edge cases and confusing patterns
        self.edge_cases = self._generate_edge_cases()
        
    def _load_real_iocs(self):
        """Load real IOCs from threat feeds."""
        logger.info("Loading real IOCs from threat intelligence feeds...")
        
        # Try to load from cache first
        cached = self.real_fetcher.load_cached(max_age_days=7)
        
        # If cache is empty, fetch fresh
        if not any(cached.values()):
            logger.info("Cache empty or outdated, fetching fresh IOCs...")
            cached = self.real_fetcher.fetch_all()
        
        # Convert sets to lists for random sampling
        for key in ['domains', 'urls', 'ips']:
            if key in cached and cached[key]:
                self.real_iocs[key] = list(cached[key])[:1000]  # Limit to 1000 each
        
        for key in ['hashes_md5', 'hashes_sha256']:
            if key in cached and cached[key]:
                self.real_iocs[key] = list(cached[key])[:500]  # Limit hashes
        
        # If still no IOCs (network issues), use fallback
        if not any(self.real_iocs.values()):
            logger.info("Using fallback IOCs due to network issues")
            self.real_iocs['domains'] = fallback_iocs.MALICIOUS_DOMAINS
            self.real_iocs['urls'] = fallback_iocs.MALICIOUS_URLS
            self.real_iocs['ips'] = fallback_iocs.MALICIOUS_IPS
            self.real_iocs['hashes_md5'] = fallback_iocs.MALWARE_MD5
            self.real_iocs['hashes_sha256'] = fallback_iocs.MALWARE_SHA256
        
        logger.info(f"Loaded {sum(len(v) for v in self.real_iocs.values())} real/fallback IOCs")
    
    def _generate_edge_cases(self) -> Dict[str, List[str]]:
        """Generate edge cases that are hard to classify."""
        return {
            'private_ips': [
                '192.168.0.1', '192.168.1.1', '192.168.1.254',
                '10.0.0.1', '10.1.1.1', '10.10.10.10',
                '172.16.0.1', '172.31.255.254',
                '127.0.0.1', '0.0.0.0', '255.255.255.255'
            ],
            'version_strings': [
                '1.2.3.4', '2.0.0.1', '3.14.15.92',
                '10.11.12.13', '99.99.99.99',
            ],
            'local_domains': [
                'localhost', 'localhost.localdomain',
                'local', 'test', 'example', 'invalid',
                'mail.local', 'server.local', 'app.test'
            ],
            'common_hashes': [
                'd41d8cd98f00b204e9800998ecf8427e',  # MD5 of empty string
                'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',  # SHA256 of empty string
                '5d41402abc4b2a76b9719d911017c592',  # MD5 of "hello"
            ],
            'ambiguous_strings': [
                'abc123def456',  # Could be partial hash
                '192.168',  # Incomplete IP
                'http://',  # Incomplete URL
                'user@',  # Incomplete email
                '.com',  # TLD only
                'CVE-',  # Incomplete CVE
            ],
            'obfuscated': [
                '192[.]168[.]1[.]1',  # Defanged IP
                'hxxp://malware[.]com',  # Defanged URL
                'malware[at]evil[dot]com',  # Defanged email
                '192.168.001.001',  # Zero-padded IP
                '0xC0A80101',  # Hex IP
                '3232235777',  # Decimal IP
            ]
        }
    
    def generate_ultra_hard_sample(self) -> SyntheticSample:
        """Generate a single ultra-hard sample."""
        
        # Choose sample type with weighted probability
        sample_type = random.choices(
            ['clean_with_edge_cases', 'real_iocs_obfuscated', 'mixed_adversarial', 
             'benign_lookalike', 'heavily_obfuscated', 'boundary_case'],
            weights=[20, 25, 20, 15, 10, 10],
            k=1
        )[0]
        
        if sample_type == 'clean_with_edge_cases':
            return self._generate_clean_with_edge_cases()
        elif sample_type == 'real_iocs_obfuscated':
            return self._generate_real_iocs_obfuscated()
        elif sample_type == 'mixed_adversarial':
            return self._generate_mixed_adversarial()
        elif sample_type == 'benign_lookalike':
            return self._generate_benign_lookalike()
        elif sample_type == 'heavily_obfuscated':
            return self._generate_heavily_obfuscated()
        else:  # boundary_case
            return self._generate_boundary_case()
    
    def _generate_clean_with_edge_cases(self) -> SyntheticSample:
        """Generate clean text with edge cases that might confuse the model."""
        base_text = self.generate_clean_text('medium')
        
        # Add some edge cases
        edge_insertions = []
        
        # Add version numbers that look like IPs
        if random.random() > 0.5:
            version = random.choice(self.edge_cases['version_strings'])
            edge_insertions.append(f"Software version {version} installed")
        
        # Add local domains
        if random.random() > 0.5:
            domain = random.choice(self.edge_cases['local_domains'])
            edge_insertions.append(f"Connected to {domain} successfully")
        
        # Add hash-like strings in benign context
        if random.random() > 0.5:
            fake_hash = ''.join(random.choices('0123456789abcdef', k=32))
            edge_insertions.append(f"Transaction ID: {fake_hash}")
        
        # Combine
        if edge_insertions:
            text = base_text + " " + " ".join(edge_insertions)
        else:
            text = base_text
        
        return SyntheticSample(
            text=text,
            labels=[],  # Clean sample
            risk_score=0.0,
            context_type='clean'
        )
    
    def _generate_real_iocs_obfuscated(self) -> SyntheticSample:
        """Generate samples with real IOCs but obfuscated."""
        if not self.use_real_iocs or not any(self.real_iocs.values()):
            # Fallback to synthetic if no real IOCs
            return self.generate_context_with_iocs(n_iocs=2, malicious_ratio=1.0)
        
        # Pick 1-3 real IOCs
        num_iocs = random.randint(1, 3)
        selected_iocs = []
        labels = []
        
        # Build context
        contexts = [
            "Security alert: Detected communication with {ioc}",
            "Malicious activity identified from {ioc}",
            "Blocked connection attempt to {ioc}",
            "Suspicious file with signature {ioc}",
            "Threat intelligence match for {ioc}",
        ]
        
        text_parts = []
        current_pos = 0
        
        for i in range(num_iocs):
            # Pick IOC type
            available_types = [k for k, v in self.real_iocs.items() if v]
            if not available_types:
                break
                
            ioc_type = random.choice(available_types)
            ioc = random.choice(self.real_iocs[ioc_type])
            
            # Maybe obfuscate
            if random.random() > 0.4:
                if 'domain' in ioc_type or 'url' in ioc_type:
                    # Defang
                    ioc = ioc.replace('.', '[.]').replace('http://', 'hxxp://').replace('https://', 'hxxps://')
                elif 'ip' in ioc_type:
                    # Defang or zero-pad
                    if random.random() > 0.5:
                        ioc = ioc.replace('.', '[.]')
                    else:
                        parts = ioc.split('.')
                        if len(parts) == 4:
                            ioc = '.'.join(f"{int(p):03d}" for p in parts if p.isdigit())
                
            # Create context
            context = random.choice(contexts).format(ioc=ioc)
            text_parts.append(context)
            
            # Track position for labels
            start = len(' '.join(text_parts[:i])) + (i > 0) + context.index(ioc)
            end = start + len(ioc)
            
            # Map to IOCType
            if 'domain' in ioc_type:
                labels.append((start, end, IOCType.DOMAIN))
            elif 'url' in ioc_type:
                labels.append((start, end, IOCType.URL))
            elif 'ip' in ioc_type:
                labels.append((start, end, IOCType.IPV4))
            elif 'md5' in ioc_type:
                labels.append((start, end, IOCType.MD5))
            elif 'sha256' in ioc_type:
                labels.append((start, end, IOCType.SHA256))
        
        text = ' '.join(text_parts)
        
        return SyntheticSample(
            text=text,
            labels=labels,
            risk_score=0.9,  # High risk for real IOCs
            context_type='malicious'
        )
    
    def _generate_mixed_adversarial(self) -> SyntheticSample:
        """Generate mixed clean and malicious with adversarial techniques."""
        # Start with clean base
        clean_text = self.generate_clean_text('short')
        
        # Add confusing negatives
        confusing = self.adversarial_gen.generate_confusing_negatives(1)[0]
        
        # Maybe add a real IOC
        labels = []
        if self.use_real_iocs and random.random() > 0.5:
            available_types = [k for k, v in self.real_iocs.items() if v]
            if available_types:
                ioc_type = random.choice(available_types)
                ioc = random.choice(self.real_iocs[ioc_type])
                
                # Generate adversarial version
                variants = self.adversarial_gen.generate_adversarial_iocs(ioc, ioc_type)
                ioc = random.choice(variants)
                
                # Insert into text
                insertion = f" Found suspicious indicator: {ioc}"
                start = len(clean_text) + len(confusing) + insertion.index(ioc)
                end = start + len(ioc)
                
                # Map to IOCType
                if 'domain' in ioc_type:
                    labels.append((start, end, IOCType.DOMAIN))
                elif 'ip' in ioc_type:
                    labels.append((start, end, IOCType.IPV4))
                
                text = clean_text + " " + confusing + insertion
            else:
                text = clean_text + " " + confusing
        else:
            text = clean_text + " " + confusing
        
        risk = 0.5 if labels else 0.1  # Low risk if no real IOCs
        
        return SyntheticSample(
            text=text,
            labels=labels,
            risk_score=risk,
            context_type='mixed'
        )
    
    def _generate_benign_lookalike(self) -> SyntheticSample:
        """Generate benign text that strongly resembles IOCs."""
        templates = [
            "Database connection string: mysql://user:pass@192.168.1.100:3306/db",
            "Git commit hash: {hash} pushed to branch main",
            "Docker image sha256:{hash} pulled successfully",
            "SSL certificate fingerprint: {hash}",
            "API endpoint: https://api.example.com/v1/users/192.168.1.1",
            "Build artifact checksum: {hash}",
            "Session token: {token}",
            "IPv4 address validation regex: ^(?:[0-9]{{1,3}}\\.)+[0-9]{{1,3}}$",  # Escaped braces
            "Test data: user@test.com, password: {hash}",
            "Configuration: server=10.0.0.1, port=8080, ssl=true",
        ]
        
        template = random.choice(templates)
        
        # Generate benign hash-like strings
        fake_hash = ''.join(random.choices('0123456789abcdef', k=random.choice([32, 40, 64])))
        fake_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        
        text = template.format(
            hash=fake_hash,
            token=fake_token
        )
        
        # No labels - this is all benign despite appearances
        return SyntheticSample(
            text=text,
            labels=[],
            risk_score=0.0,
            context_type='clean'
        )
    
    def _generate_heavily_obfuscated(self) -> SyntheticSample:
        """Generate heavily obfuscated malicious IOCs."""
        obfuscations = [
            # URL encoding
            "https%3A%2F%2Fmalware%2Eevil%2Ecom%2Fpayload",
            # Base64 partial
            "aHR0cHM6Ly9tYWx3YXJlLmV2aWwuY29t",
            # Hex encoding
            "\\x68\\x74\\x74\\x70\\x3a\\x2f\\x2f",
            # Unicode escapes
            "\\u0068\\u0074\\u0074\\u0070",
            # ROT13
            "uggc://znyjner.pbz",
            # Decimal IP
            "3232235777",  # 192.168.1.1 in decimal
            # Hex IP
            "0xC0A80101",  # 192.168.1.1 in hex
        ]
        
        text_parts = ["Detected obfuscated indicators:"]
        labels = []
        
        num_obfuscated = random.randint(1, 3)
        for _ in range(num_obfuscated):
            obf = random.choice(obfuscations)
            start = len(' '.join(text_parts)) + 1
            end = start + len(obf)
            text_parts.append(obf)
            
            # These are hard to classify correctly
            # Mark as URL or IP based on pattern
            if 'http' in obf.lower() or '%2F' in obf:
                labels.append((start, end, IOCType.URL))
            elif any(c in obf for c in ['C0A8', '3232', '0x']):
                labels.append((start, end, IOCType.IPV4))
        
        text = ' '.join(text_parts)
        
        return SyntheticSample(
            text=text,
            labels=labels,
            risk_score=0.95,  # Very high risk
            context_type='malicious'
        )
    
    def _generate_boundary_case(self) -> SyntheticSample:
        """Generate boundary cases that are ambiguous."""
        cases = [
            # Incomplete IOCs
            ("Partial IP detected: 192.168", [], 0.3),
            ("Truncated hash: 5d41402abc4b2a76b9719d91", [], 0.4),
            ("Incomplete URL: http://malware", [], 0.5),
            
            # Almost-valid IOCs
            ("IP address: 256.256.256.256", [], 0.2),  # Invalid octets
            ("Domain: this-is-a-very-long-domain-name-that-exceeds-limits.com", [], 0.3),
            ("Hash: ZZZZ402abc4b2a76b9719d911017c592", [], 0.1),  # Invalid hex
            
            # Context-dependent
            ("CVE-2024-PENDING not yet assigned", [], 0.1),
            ("Testing with dummy IP 999.999.999.999", [], 0.0),
            ("Localhost connection on 127.0.0.1 is safe", [], 0.0),
        ]
        
        text, labels, risk = random.choice(cases)
        
        # Sometimes add a real IOC to make it harder
        if random.random() > 0.7 and self.use_real_iocs:
            available_types = [k for k, v in self.real_iocs.items() if v]
            if available_types:
                ioc_type = random.choice(available_types)
                ioc = random.choice(self.real_iocs[ioc_type])
                addition = f" Also found: {ioc}"
                start = len(text) + addition.index(ioc)
                end = start + len(ioc)
                
                if 'ip' in ioc_type:
                    labels.append((start, end, IOCType.IPV4))
                    risk = min(risk + 0.5, 1.0)
                
                text += addition
        
        return SyntheticSample(
            text=text,
            labels=labels,
            risk_score=risk,
            context_type='boundary'
        )
    
    def generate_ultra_hard_batch(self, 
                                 n_samples: int = 1000,
                                 clean_ratio: float = 0.3,
                                 easy_ratio: float = 0.1,
                                 hard_ratio: float = 0.6) -> List[SyntheticSample]:
        """Generate a batch of ultra-hard training samples.
        
        Args:
            n_samples: Total samples to generate
            clean_ratio: Ratio of truly clean samples (but with edge cases)
            easy_ratio: Ratio of clearly malicious samples
            hard_ratio: Ratio of hard/ambiguous samples
        """
        samples = []
        
        n_clean = int(n_samples * clean_ratio)
        n_easy = int(n_samples * easy_ratio)
        n_hard = n_samples - n_clean - n_easy
        
        logger.info(f"Generating ultra-hard dataset:")
        logger.info(f"  - Clean (with edge cases): {n_clean}")
        logger.info(f"  - Easy malicious: {n_easy}")
        logger.info(f"  - Hard/ambiguous: {n_hard}")
        
        # Generate clean samples with edge cases
        for _ in range(n_clean):
            sample = self._generate_clean_with_edge_cases() if random.random() > 0.3 else self._generate_benign_lookalike()
            samples.append(sample)
        
        # Generate easy malicious samples (clear IOCs)
        for _ in range(n_easy):
            if self.use_real_iocs and self.real_iocs:
                sample = self._generate_real_iocs_obfuscated()
            else:
                sample = self.generate_context_with_iocs(n_iocs=random.randint(2, 4), malicious_ratio=1.0)
            samples.append(sample)
        
        # Generate hard samples
        for _ in range(n_hard):
            sample = self.generate_ultra_hard_sample()
            samples.append(sample)
        
        random.shuffle(samples)
        return samples