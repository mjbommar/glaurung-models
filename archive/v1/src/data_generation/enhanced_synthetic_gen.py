#!/usr/bin/env python3
"""Enhanced synthetic data generation with balanced clean/suspicious samples."""

import random
import string
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .ioc_patterns import IOCType, SUSPICIOUS_TLDS, MALWARE_FAMILIES
from .synthetic_gen import IOCSyntheticGenerator, SyntheticSample


class EnhancedSyntheticGenerator(IOCSyntheticGenerator):
    """Enhanced generator with better clean/suspicious balance."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.clean_text_sources = self._load_clean_text_sources()
        
    def _load_clean_text_sources(self) -> Dict[str, List[str]]:
        """Load various sources of clean text."""
        sources = {
            'technical': [],
            'business': [],
            'general': [],
            'code': []
        }
        
        # Try to load from system dictionary
        dict_paths = [
            '/usr/share/dict/words',
            '/usr/dict/words',
            '/usr/share/dict/american-english'
        ]
        
        words = []
        for path in dict_paths:
            if Path(path).exists():
                with open(path, 'r') as f:
                    words = [w.strip() for w in f.readlines() if len(w.strip()) > 3][:10000]
                    break
        
        if not words:
            # Fallback to basic word list
            words = self._generate_word_list()
        
        # Technical documentation patterns
        sources['technical'] = [
            "The {component} service processes {number} requests per second with {metric}% efficiency.",
            "Database replication completed successfully with {number} records synchronized.",
            "System performance metrics show {metric}% CPU utilization during peak hours.",
            "Memory allocation for {component} module optimized to {number}MB.",
            "Network latency between {location1} and {location2} averages {number}ms.",
            "Backup process completed in {number} minutes with {metric}% compression ratio.",
            "Load balancer distributing traffic across {number} healthy instances.",
            "Cache hit ratio improved to {metric}% after optimization.",
            "Application response time reduced by {metric}% through query optimization.",
            "Monitoring dashboard shows all {number} services operational.",
        ]
        
        # Business communications
        sources['business'] = [
            "Q{quarter} revenue increased by {metric}% year-over-year to ${number}M.",
            "Team completed {number} projects ahead of schedule this quarter.",
            "Customer satisfaction score improved to {metric}% based on {number} responses.",
            "New product launch attracted {number} users in the first week.",
            "Operating costs reduced by {metric}% through process automation.",
            "Employee retention rate stands at {metric}% for the fiscal year.",
            "Market share grew by {metric} basis points to {number}%.",
            "Digital transformation initiative {metric}% complete with {number} systems migrated.",
            "Supply chain efficiency improved by {metric}% reducing delivery time to {number} days.",
            "Innovation lab filed {number} new patents this year.",
        ]
        
        # General text patterns
        sources['general'] = [
            "The weather forecast predicts {weather} conditions with temperatures around {number} degrees.",
            "Local news reports {number} new businesses opened in the downtown area.",
            "Scientific study involving {number} participants showed {metric}% improvement.",
            "Historical records indicate the building was constructed in {year}.",
            "Educational program enrolled {number} students this semester.",
            "Transportation infrastructure project {metric}% complete after {number} months.",
            "Environmental impact assessment shows {metric}% reduction in emissions.",
            "Community event attracted over {number} attendees last weekend.",
            "Library catalog expanded by {number} new titles this month.",
            "Research paper cites {number} references from peer-reviewed journals.",
        ]
        
        # Code comments and logs
        sources['code'] = [
            "# TODO: Refactor {function} to improve performance by {metric}%",
            "// Fixed bug in {module} that caused {number} test failures",
            "/* Deprecated: Use {newfunction} instead for better compatibility */",
            "DEBUG: Processing batch {number} of {total} items",
            "INFO: Configuration loaded from {configfile} successfully",
            "WARNING: Memory usage at {metric}% of allocated limit",
            "# Optimization: Reduced complexity from O(n^2) to O(n log n)",
            "// Version {version} released with {number} bug fixes",
            "/* Performance benchmark: {number} operations per second */",
            "TRACE: Method {function} executed in {number}ms",
        ]
        
        sources['words'] = words
        return sources
    
    def _generate_word_list(self) -> List[str]:
        """Generate a basic word list if system dictionary not available."""
        return [
            "system", "process", "application", "service", "database", "network",
            "server", "client", "module", "component", "interface", "protocol",
            "function", "method", "class", "object", "variable", "constant",
            "configuration", "settings", "parameters", "options", "preferences",
            "user", "admin", "manager", "developer", "engineer", "analyst",
            "data", "information", "record", "document", "report", "analysis",
            "performance", "optimization", "efficiency", "reliability", "security",
            "backup", "restore", "recovery", "redundancy", "failover", "disaster",
            "monitor", "alert", "notification", "warning", "error", "exception",
            "request", "response", "transaction", "operation", "execution", "completion",
            "input", "output", "stream", "buffer", "cache", "memory", "storage",
            "update", "upgrade", "patch", "release", "version", "build",
            "test", "debug", "trace", "log", "audit", "review"
        ]
    
    def generate_clean_text(self, length: str = 'medium') -> str:
        """Generate completely clean text with NO IOCs."""
        
        # Choose text type
        text_type = random.choice(['technical', 'business', 'general', 'code'])
        template = random.choice(self.clean_text_sources[text_type])
        
        # Fill in placeholders with safe values
        replacements = {
            '{component}': random.choice(['authentication', 'logging', 'caching', 'routing', 'scheduling']),
            '{module}': random.choice(['core', 'utils', 'helpers', 'middleware', 'services']),
            '{function}': random.choice(['processData', 'validateInput', 'handleRequest', 'updateCache']),
            '{newfunction}': random.choice(['processDataV2', 'validateInputSecure', 'handleRequestAsync']),
            '{configfile}': random.choice(['config.json', 'settings.yaml', 'app.properties', 'environment.env']),
            '{number}': str(random.randint(1, 9999)),
            '{total}': str(random.randint(100, 10000)),
            '{metric}': str(round(random.uniform(10, 99), 1)),
            '{quarter}': str(random.randint(1, 4)),
            '{year}': str(random.randint(1900, 2020)),
            '{version}': f"{random.randint(1, 9)}.{random.randint(0, 99)}.{random.randint(0, 999)}",
            '{weather}': random.choice(['sunny', 'cloudy', 'rainy', 'partly cloudy', 'clear']),
            '{location1}': random.choice(['datacenter A', 'region US-East', 'primary site', 'zone 1']),
            '{location2}': random.choice(['datacenter B', 'region EU-West', 'backup site', 'zone 2']),
        }
        
        text = template
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # Add some random safe words if needed for length
        if length == 'long':
            words = random.sample(self.clean_text_sources['words'], min(20, len(self.clean_text_sources['words'])))
            additional = " Additional context: " + " ".join(words[:random.randint(5, 15)])
            text += additional
        
        return text
    
    def generate_mixed_text_with_iocs(self, num_iocs: int = 1) -> SyntheticSample:
        """Generate text that mixes clean content with IOCs."""
        
        # Start with clean base text
        base_text = self.generate_clean_text('medium')
        
        # IOC insertion templates
        ioc_insertions = [
            " Detected suspicious activity from {ioc}.",
            " Connection established to {ioc}.",
            " File hash {ioc} identified.",
            " Alert triggered for {ioc}.",
            " Scanning revealed {ioc}.",
            " Found reference to {ioc} in logs.",
            " Unusual pattern matching {ioc}.",
            " Security event involving {ioc}.",
        ]
        
        labels = []
        
        # Add IOCs
        for i in range(num_iocs):
            ioc_type = random.choice([
                'ipv4', 'ipv6', 'domain', 'url', 'md5', 'sha256', 'email', 'cve'
            ])
            
            # Generate IOC
            if ioc_type == 'ipv4':
                ioc = self.generate_ipv4(malicious=random.choice([True, False]))
                ioc_enum = IOCType.IPV4
            elif ioc_type == 'ipv6':
                ioc = self.generate_ipv6()
                ioc_enum = IOCType.IPV6
            elif ioc_type == 'domain':
                ioc = self.generate_domain(malicious=random.choice([True, False]))
                ioc_enum = IOCType.DOMAIN
            elif ioc_type == 'url':
                ioc = self.generate_url(malicious=random.choice([True, False]))
                ioc_enum = IOCType.URL
            elif ioc_type == 'md5':
                ioc = self.generate_hash('md5')
                ioc_enum = IOCType.MD5
            elif ioc_type == 'sha256':
                ioc = self.generate_hash('sha256')
                ioc_enum = IOCType.SHA256
            elif ioc_type == 'email':
                ioc = self.generate_email(malicious=random.choice([True, False]))
                ioc_enum = IOCType.EMAIL
            elif ioc_type == 'cve':
                ioc = self.generate_cve()
                ioc_enum = IOCType.CVE
            else:
                continue
            
            # Insert IOC into text
            insertion = random.choice(ioc_insertions).format(ioc=ioc)
            start_pos = len(base_text) + insertion.index(ioc)
            end_pos = start_pos + len(ioc)
            
            base_text += insertion
            labels.append((start_pos, end_pos, ioc_enum))
        
        # Calculate risk based on number and type of IOCs
        risk_score = min(1.0, num_iocs * 0.3)
        
        return SyntheticSample(
            text=base_text,
            labels=labels,
            risk_score=risk_score,
            context_type='suspicious' if num_iocs > 0 else 'clean'
        )
    
    def generate_ipv6(self) -> str:
        """Generate IPv6 address."""
        # Generate 8 groups of 4 hex digits
        groups = []
        for _ in range(8):
            group = ''.join(random.choices('0123456789abcdef', k=4))
            groups.append(group)
        return ':'.join(groups)
    
    def generate_balanced_batch(self, 
                               n_samples: int = 1000,
                               clean_ratio: float = 0.4,
                               light_suspicious: float = 0.3,
                               heavy_suspicious: float = 0.3) -> List[SyntheticSample]:
        """Generate balanced batch with proper distribution.
        
        Args:
            n_samples: Total samples to generate
            clean_ratio: Ratio of completely clean samples (0 IOCs)
            light_suspicious: Ratio of lightly suspicious (1-2 IOCs)
            heavy_suspicious: Ratio of heavily suspicious (3+ IOCs)
        """
        samples = []
        
        n_clean = int(n_samples * clean_ratio)
        n_light = int(n_samples * light_suspicious)
        n_heavy = n_samples - n_clean - n_light
        
        print(f"Generating balanced dataset:")
        print(f"  - Clean (0 IOCs): {n_clean}")
        print(f"  - Light suspicious (1-2 IOCs): {n_light}")
        print(f"  - Heavy suspicious (3+ IOCs): {n_heavy}")
        
        # Generate clean samples
        for _ in range(n_clean):
            text = self.generate_clean_text(random.choice(['short', 'medium', 'long']))
            samples.append(SyntheticSample(
                text=text,
                labels=[],
                risk_score=0.0,
                context_type='clean'
            ))
        
        # Generate lightly suspicious samples
        for _ in range(n_light):
            num_iocs = random.randint(1, 2)
            samples.append(self.generate_mixed_text_with_iocs(num_iocs))
        
        # Generate heavily suspicious samples
        for _ in range(n_heavy):
            num_iocs = random.randint(3, 6)
            # Use original method for heavy suspicious
            samples.append(self.generate_context_with_iocs(
                n_iocs=num_iocs,
                malicious_ratio=0.8
            ))
        
        random.shuffle(samples)
        return samples