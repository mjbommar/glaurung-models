#!/usr/bin/env python3
"""Fetch and process real IOCs from threat intelligence feeds."""

import requests
import csv
import json
import re
import logging
from typing import List, Dict, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import random
from io import StringIO

logger = logging.getLogger(__name__)


class RealIOCFetcher:
    """Fetch real IOCs from various threat intelligence feeds."""
    
    def __init__(self, cache_dir: str = "data/real_iocs"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feeds to fetch from
        self.feeds = {
            'urlhaus_recent': 'https://urlhaus.abuse.ch/downloads/csv_recent/',
            'urlhaus_online': 'https://urlhaus.abuse.ch/downloads/csv_online/',
            'feodotracker_ipblocklist': 'https://feodotracker.abuse.ch/downloads/ipblocklist_recommended.txt',
            'sslbl_ipblacklist': 'https://sslbl.abuse.ch/blacklist/sslipblacklist.txt',
            'threatfox_recent': 'https://threatfox.abuse.ch/export/csv/recent/',
        }
        
        # Additional GitHub-hosted IOC lists (with fallbacks)
        self.github_iocs = {
            'fireeye_sunburst': 'https://raw.githubusercontent.com/fireeye/sunburst_countermeasures/main/all-hashes.txt',
            'botvrij_iocs': 'https://www.botvrij.eu/data/ioclist.domain.raw',
        }
        
        self.iocs = {
            'domains': set(),
            'urls': set(),
            'ips': set(),
            'hashes_md5': set(),
            'hashes_sha256': set(),
            'emails': set(),
        }
        
    def fetch_urlhaus(self) -> Dict[str, Set[str]]:
        """Fetch malicious URLs from URLhaus."""
        logger.info("Fetching URLhaus recent malicious URLs...")
        
        try:
            response = requests.get(
                self.feeds['urlhaus_recent'],
                headers={'User-Agent': 'IOC-Fetcher/1.0'},
                timeout=30,
                verify=False  # Skip SSL verification for abuse.ch
            )
            response.raise_for_status()
            
            # Skip comment lines and parse CSV
            lines = [line for line in response.text.splitlines() if not line.startswith('#')]
            csv_data = StringIO('\n'.join(lines))
            reader = csv.DictReader(csv_data)
            
            urls = set()
            domains = set()
            
            for row in reader:
                if 'url' in row and row['url']:
                    url = row['url'].strip('"')
                    urls.add(url)
                    
                    # Extract domain from URL
                    domain_match = re.search(r'https?://([^/]+)', url)
                    if domain_match:
                        domain = domain_match.group(1)
                        domains.add(domain)
            
            logger.info(f"Fetched {len(urls)} malicious URLs and {len(domains)} domains from URLhaus")
            return {'urls': urls, 'domains': domains}
            
        except Exception as e:
            logger.error(f"Failed to fetch URLhaus: {e}")
            return {'urls': set(), 'domains': set()}
    
    def fetch_feodotracker(self) -> Set[str]:
        """Fetch Feodo Tracker botnet C2 IPs."""
        logger.info("Fetching Feodo Tracker C2 IPs...")
        
        try:
            response = requests.get(
                self.feeds['feodotracker_ipblocklist'],
                headers={'User-Agent': 'IOC-Fetcher/1.0'},
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            
            ips = set()
            for line in response.text.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Validate IP
                    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', line):
                        ips.add(line)
            
            logger.info(f"Fetched {len(ips)} C2 IPs from Feodo Tracker")
            return ips
            
        except Exception as e:
            logger.error(f"Failed to fetch Feodo Tracker: {e}")
            return set()
    
    def fetch_fireeye_hashes(self) -> Dict[str, Set[str]]:
        """Fetch FireEye tool hashes or generate sample hashes."""
        logger.info("Fetching malware hashes...")
        
        # Try multiple sources
        hash_sources = [
            self.github_iocs.get('fireeye_sunburst', ''),
            'https://raw.githubusercontent.com/Neo23x0/signature-base/master/iocs/hash-iocs.txt',
        ]
        
        md5_hashes = set()
        sha256_hashes = set()
        
        for source_url in hash_sources:
            if not source_url:
                continue
            try:
                response = requests.get(
                    source_url,
                    headers={'User-Agent': 'IOC-Fetcher/1.0'},
                    timeout=10,
                    verify=False
                )
                response.raise_for_status()
                
                for line in response.text.splitlines():
                    line = line.strip()
                    if len(line) == 32 and all(c in '0123456789abcdefABCDEF' for c in line):
                        md5_hashes.add(line.lower())
                    elif len(line) == 64 and all(c in '0123456789abcdefABCDEF' for c in line):
                        sha256_hashes.add(line.lower())
                
                if md5_hashes or sha256_hashes:
                    break  # Found hashes, stop trying other sources
                    
            except Exception as e:
                logger.debug(f"Failed to fetch from {source_url}: {e}")
                continue
        
        # If no real hashes found, generate some common malware hashes as examples
        if not md5_hashes and not sha256_hashes:
            logger.info("Using sample malware hashes for training")
            md5_hashes = {
                '44d88612fea8a8f36de82e1278abb02f',  # EICAR test file
                '098f6bcd4621d373cade4e832627b4f6',  # test
                '5d41402abc4b2a76b9719d911017c592',  # hello
                'e10adc3949ba59abbe56e057f20f883e',  # 123456
            }
            sha256_hashes = {
                '275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f',  # EICAR
                '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',  # test
            }
        
        logger.info(f"Total hashes: {len(md5_hashes)} MD5 and {len(sha256_hashes)} SHA256")
        return {'md5': md5_hashes, 'sha256': sha256_hashes}
    
    def fetch_all(self) -> Dict[str, Set[str]]:
        """Fetch IOCs from all configured sources."""
        logger.info("Fetching IOCs from all sources...")
        
        # URLhaus
        urlhaus_data = self.fetch_urlhaus()
        self.iocs['urls'].update(urlhaus_data.get('urls', set()))
        self.iocs['domains'].update(urlhaus_data.get('domains', set()))
        
        # Feodo Tracker
        feodo_ips = self.fetch_feodotracker()
        self.iocs['ips'].update(feodo_ips)
        
        # FireEye hashes
        fireeye_hashes = self.fetch_fireeye_hashes()
        self.iocs['hashes_md5'].update(fireeye_hashes.get('md5', set()))
        self.iocs['hashes_sha256'].update(fireeye_hashes.get('sha256', set()))
        
        # Save to cache
        self.save_cache()
        
        logger.info(f"Total IOCs collected:")
        logger.info(f"  - URLs: {len(self.iocs['urls'])}")
        logger.info(f"  - Domains: {len(self.iocs['domains'])}")
        logger.info(f"  - IPs: {len(self.iocs['ips'])}")
        logger.info(f"  - MD5 hashes: {len(self.iocs['hashes_md5'])}")
        logger.info(f"  - SHA256 hashes: {len(self.iocs['hashes_sha256'])}")
        
        return self.iocs
    
    def save_cache(self):
        """Save fetched IOCs to cache files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for ioc_type, ioc_set in self.iocs.items():
            if ioc_set:
                cache_file = self.cache_dir / f"{ioc_type}_{timestamp}.txt"
                with open(cache_file, 'w') as f:
                    for ioc in sorted(ioc_set):
                        f.write(f"{ioc}\n")
                logger.info(f"Saved {len(ioc_set)} {ioc_type} to {cache_file}")
    
    def load_cached(self, max_age_days: int = 7) -> Dict[str, Set[str]]:
        """Load IOCs from cache if available and recent."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for ioc_type in self.iocs.keys():
            # Find most recent cache file
            pattern = f"{ioc_type}_*.txt"
            cache_files = list(self.cache_dir.glob(pattern))
            
            if cache_files:
                # Sort by modification time
                cache_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest_file = cache_files[0]
                
                # Check if recent enough
                file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                if file_time > cutoff_date:
                    with open(latest_file, 'r') as f:
                        iocs = {line.strip() for line in f if line.strip()}
                        self.iocs[ioc_type].update(iocs)
                        logger.info(f"Loaded {len(iocs)} cached {ioc_type} from {latest_file}")
        
        return self.iocs
    
    def get_random_iocs(self, n_samples: int = 100) -> Dict[str, List[str]]:
        """Get random samples of real IOCs."""
        samples = {}
        
        for ioc_type, ioc_set in self.iocs.items():
            if ioc_set:
                sample_size = min(n_samples, len(ioc_set))
                samples[ioc_type] = random.sample(list(ioc_set), sample_size)
        
        return samples


class AdversarialIOCGenerator:
    """Generate adversarial and edge case IOCs that are harder to classify."""
    
    def __init__(self):
        # Common benign patterns that look like IOCs
        self.benign_lookalikes = {
            'version_numbers': [
                '192.168.1.1',  # Looks like IP but common in docs
                '10.0.0.1',
                '127.0.0.1',
                '3.14.159.265',  # Pi
                '2.71.828.182',  # e
            ],
            'product_codes': [
                'abc123def456789012345678901234567',  # 32 chars, looks like MD5
                'XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX',  # UUID-like
            ],
            'common_domains': [
                'localhost.localdomain',
                'example.com',
                'test.local',
                'www.google.com',
                'github.com',
            ],
        }
        
        # Obfuscation techniques
        self.obfuscations = {
            'url_encoding': lambda s: ''.join(f'%{ord(c):02x}' if random.random() > 0.7 else c for c in s),
            'base64_partial': lambda s: s[:len(s)//2] + '==' + s[len(s)//2:],
            'zero_width': lambda s: s[:len(s)//2] + '\u200b' + s[len(s)//2:],  # Zero-width space
            'homoglyphs': self._apply_homoglyphs,
            'case_variation': lambda s: ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in s),
        }
    
    def _apply_homoglyphs(self, text: str) -> str:
        """Replace characters with similar-looking ones."""
        homoglyphs = {
            'o': ['0', 'О', 'о'],  # Latin o, zero, Cyrillic O
            'O': ['0', 'О'],
            'l': ['1', 'I', '|', 'і'],  # Latin l, one, uppercase i, pipe, Cyrillic i
            'a': ['а', '@'],  # Latin a, Cyrillic a
            'e': ['е', '3'],  # Latin e, Cyrillic e
            'i': ['і', '1'],  # Latin i, Cyrillic i
        }
        
        result = []
        for char in text:
            if char in homoglyphs and random.random() > 0.7:
                result.append(random.choice(homoglyphs[char]))
            else:
                result.append(char)
        return ''.join(result)
    
    def generate_confusing_negatives(self, n_samples: int = 100) -> List[str]:
        """Generate benign text that looks like it might contain IOCs."""
        samples = []
        
        templates = [
            "Software version {version} installed successfully",
            "Connection pool size set to {number}",
            "Hash table initialized with {number} buckets",
            "IPv6 support enabled in configuration",
            "Certificate SHA fingerprint updated",
            "Domain controller synchronization completed",
            "URL rewriting rules applied",
            "Binary checksum validation passed",
            "Registry cleanup removed {number} entries",
            "Email validation regex updated",
            "Network interface {iface} configured",
            "Cryptographic salt generated: {salt}",
            "Session token refreshed for user{number}",
            "API endpoint /api/v{version}/users registered",
            "Database connection string updated",
        ]
        
        for _ in range(n_samples):
            template = random.choice(templates)
            text = template.format(
                version=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                number=random.randint(1, 65535),
                iface=random.choice(['eth0', 'wlan0', 'lo', 'docker0']),
                salt=''.join(random.choices('0123456789abcdef', k=32)),
            )
            samples.append(text)
        
        return samples
    
    def generate_adversarial_iocs(self, base_ioc: str, ioc_type: str) -> List[str]:
        """Generate adversarial versions of an IOC."""
        variants = [base_ioc]  # Include original
        
        if ioc_type == 'domain':
            # Domain variations
            variants.append(base_ioc.replace('.', '[.]'))  # Defanged
            variants.append(base_ioc.replace('.', '[dot]'))
            variants.append(base_ioc.replace('.', '_dot_'))
            variants.append('hxxp://' + base_ioc)  # Defanged protocol
            variants.append(self.obfuscations['homoglyphs'](base_ioc))
            
        elif ioc_type == 'ip':
            # IP obfuscations
            parts = base_ioc.split('.')
            if len(parts) == 4:
                # Octal representation
                octal = '.'.join(f"0{int(p):o}" for p in parts)
                variants.append(octal)
                # Zero-padded
                padded = '.'.join(f"{int(p):03d}" for p in parts)
                variants.append(padded)
                # Defanged
                variants.append(base_ioc.replace('.', '[.]'))
                
        elif ioc_type in ['md5', 'sha256']:
            # Hash variations
            variants.append(base_ioc.upper())
            variants.append(base_ioc.lower())
            # Partial with ellipsis
            variants.append(base_ioc[:8] + '...' + base_ioc[-8:])
            # With spaces
            variants.append(' '.join(base_ioc[i:i+4] for i in range(0, len(base_ioc), 4)))
        
        return variants
    
    def generate_mixed_context(self, real_iocs: Dict[str, List[str]], clean_text: str) -> str:
        """Mix real IOCs into clean text in subtle ways."""
        # Randomly insert IOCs into clean text
        words = clean_text.split()
        
        # Insert 1-3 IOCs at random positions
        num_insertions = random.randint(1, 3)
        
        for _ in range(num_insertions):
            if random.random() > 0.5 and real_iocs:
                # Pick a random IOC type
                available_types = [k for k, v in real_iocs.items() if v]
                if available_types:
                    ioc_type = random.choice(available_types)
                    ioc = random.choice(real_iocs[ioc_type])
                    
                    # Insert at random position
                    position = random.randint(0, len(words))
                    
                    # Sometimes obfuscate
                    if random.random() > 0.6:
                        obfuscation = random.choice(list(self.obfuscations.values()))
                        ioc = obfuscation(ioc)
                    
                    words.insert(position, ioc)
        
        return ' '.join(words)