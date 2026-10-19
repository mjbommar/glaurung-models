"""Synthetic IOC data generation module."""

import random
import string
import ipaddress
from typing import List, Dict, Tuple, Optional
import hashlib
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

from .ioc_patterns import IOCType, SUSPICIOUS_TLDS, MALWARE_FAMILIES, SUSPICIOUS_PORTS


@dataclass
class SyntheticSample:
    """Synthetic data sample."""
    text: str
    labels: List[Tuple[int, int, IOCType]]  # (start, end, type)
    risk_score: float
    context_type: str  # 'malicious', 'benign', 'mixed'


class IOCSyntheticGenerator:
    """Generate synthetic IOC training data."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed:
            random.seed(seed)
        
        self.benign_words = self._load_benign_vocabulary()
        self.threat_contexts = self._load_threat_contexts()
        
    def _load_benign_vocabulary(self) -> List[str]:
        """Load common benign words for context generation."""
        return [
            "system", "user", "file", "process", "application", "service",
            "update", "download", "install", "configure", "network", "server",
            "client", "database", "connection", "request", "response", "data",
            "information", "security", "policy", "management", "monitoring"
        ]
    
    def _load_threat_contexts(self) -> List[str]:
        """Load threat-related context templates."""
        return [
            "Detected malicious activity from IP {ip} on port {port}",
            "Suspicious file {hash} downloaded from {domain}",
            "Command and control server at {url} contacted",
            "Registry key {registry} modified by unknown process",
            "Phishing email received from {email} containing {url}",
            "Vulnerability {cve} exploited on system {ip}",
            "Ransomware {malware} detected with hash {hash}",
            "Data exfiltration to {domain} via port {port}",
            "Suspicious PowerShell execution from {filepath}",
            "Lateral movement detected from {ip} to {ip}"
        ]
    
    def generate_ipv4(self, malicious: bool = False) -> str:
        """Generate IPv4 address."""
        if malicious:
            # Common malicious IP ranges
            octets = random.choice([
                [185, random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
                [45, random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
                [23, random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
            ])
        else:
            # Private/local ranges
            octets = random.choice([
                [192, 168, random.randint(0, 255), random.randint(0, 255)],
                [10, random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
                [172, random.randint(16, 31), random.randint(0, 255), random.randint(0, 255)],
            ])
        return ".".join(map(str, octets))
    
    def generate_domain(self, malicious: bool = False) -> str:
        """Generate domain name."""
        if malicious:
            # DGA-like domain
            length = random.randint(8, 20)
            name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
            tld = random.choice(list(SUSPICIOUS_TLDS) + ['.com', '.net', '.org'])
        else:
            # Benign-looking domain
            words = random.sample(self.benign_words, random.randint(1, 2))
            name = '-'.join(words)
            tld = random.choice(['.com', '.org', '.net', '.io', '.dev'])
        
        return f"{name}{tld}"
    
    def generate_url(self, malicious: bool = False) -> str:
        """Generate URL."""
        protocol = "https" if not malicious else random.choice(["http", "https"])
        domain = self.generate_domain(malicious)
        
        if malicious:
            # Suspicious paths
            path = random.choice([
                "/admin/config.php",
                "/wordpress/wp-admin/",
                "/phpmyadmin/",
                "/shell.aspx",
                "/upload/evil.exe",
                f"/{random.randint(1, 999999)}.bin",
                "/gate.php",
                "/panel/"
            ])
        else:
            # Normal paths
            path = random.choice([
                "/index.html",
                "/about",
                "/contact",
                "/api/v1/users",
                "/static/css/main.css",
                "/images/logo.png"
            ])
        
        return f"{protocol}://{domain}{path}"
    
    def generate_hash(self, hash_type: str = "md5") -> str:
        """Generate hash string."""
        lengths = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}
        length = lengths.get(hash_type, 32)
        return ''.join(random.choices('0123456789abcdef', k=length))
    
    def generate_email(self, malicious: bool = False) -> str:
        """Generate email address."""
        if malicious:
            # Suspicious email patterns
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 15)))
            domain = self.generate_domain(malicious=True)
        else:
            first = random.choice(["john", "jane", "admin", "info", "support", "contact"])
            last = random.choice(["smith", "doe", "johnson", "williams", "brown"])
            domain = random.choice(["gmail.com", "outlook.com", "company.com", "example.org"])
            username = f"{first}.{last}"
        
        return f"{username}@{domain}"
    
    def generate_cve(self) -> str:
        """Generate CVE identifier."""
        year = random.randint(2010, 2025)
        number = random.randint(1000, 99999)
        return f"CVE-{year}-{number}"
    
    def generate_filepath(self, windows: bool = True, malicious: bool = False) -> str:
        """Generate file path."""
        if windows:
            drive = random.choice(['C:', 'D:', 'E:'])
            if malicious:
                paths = [
                    f"{drive}\\Windows\\Temp\\{self._random_filename()}.exe",
                    f"{drive}\\Users\\Public\\{self._random_filename()}.dll",
                    f"{drive}\\ProgramData\\{self._random_filename()}\\evil.bat",
                    f"{drive}\\Windows\\System32\\{self._random_filename()}.vbs",
                ]
            else:
                paths = [
                    f"{drive}\\Program Files\\Microsoft Office\\office.exe",
                    f"{drive}\\Users\\User\\Documents\\report.docx",
                    f"{drive}\\Windows\\System32\\notepad.exe",
                ]
            return random.choice(paths)
        else:
            if malicious:
                paths = [
                    f"/tmp/.{self._random_filename()}",
                    f"/var/tmp/{self._random_filename()}.sh",
                    f"/dev/shm/{self._random_filename()}",
                    f"/usr/bin/.{self._random_filename()}"
                ]
            else:
                paths = [
                    "/usr/bin/python3",
                    "/home/user/documents/file.txt",
                    "/var/log/syslog",
                    "/etc/nginx/nginx.conf"
                ]
            return random.choice(paths)
    
    def generate_registry_key(self, malicious: bool = False) -> str:
        """Generate Windows registry key."""
        hive = random.choice(['HKEY_LOCAL_MACHINE', 'HKEY_CURRENT_USER', 'HKEY_CLASSES_ROOT'])
        
        if malicious:
            paths = [
                f"{hive}\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\{self._random_filename()}",
                f"{hive}\\System\\CurrentControlSet\\Services\\{self._random_filename()}",
                f"{hive}\\Software\\Classes\\{self._random_filename()}",
            ]
        else:
            paths = [
                f"{hive}\\Software\\Microsoft\\Windows\\CurrentVersion",
                f"{hive}\\System\\CurrentControlSet\\Control",
                f"{hive}\\Software\\Classes\\.txt",
            ]
        return random.choice(paths)
    
    def _random_filename(self) -> str:
        """Generate random filename."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 12)))
    
    def generate_context_with_iocs(self, n_iocs: int = 3, malicious_ratio: float = 0.7) -> SyntheticSample:
        """Generate text with embedded IOCs."""
        template = random.choice(self.threat_contexts)
        replacements = {}
        labels = []
        
        # Determine which IOCs to include
        ioc_types = random.sample(list(IOCType), min(n_iocs, len(IOCType) - 1))  # Exclude BENIGN
        
        for placeholder in ['{ip}', '{domain}', '{url}', '{hash}', '{email}', '{cve}', 
                           '{filepath}', '{registry}', '{port}', '{malware}']:
            if placeholder in template:
                is_malicious = random.random() < malicious_ratio
                
                if placeholder == '{ip}':
                    value = self.generate_ipv4(is_malicious)
                    ioc_type = IOCType.IPV4
                elif placeholder == '{domain}':
                    value = self.generate_domain(is_malicious)
                    ioc_type = IOCType.DOMAIN
                elif placeholder == '{url}':
                    value = self.generate_url(is_malicious)
                    ioc_type = IOCType.URL
                elif placeholder == '{hash}':
                    hash_type = random.choice(['md5', 'sha256'])
                    value = self.generate_hash(hash_type)
                    ioc_type = IOCType.MD5 if hash_type == 'md5' else IOCType.SHA256
                elif placeholder == '{email}':
                    value = self.generate_email(is_malicious)
                    ioc_type = IOCType.EMAIL
                elif placeholder == '{cve}':
                    value = self.generate_cve()
                    ioc_type = IOCType.CVE
                elif placeholder == '{filepath}':
                    value = self.generate_filepath(malicious=is_malicious)
                    ioc_type = IOCType.FILE_PATH_WIN
                elif placeholder == '{registry}':
                    value = self.generate_registry_key(is_malicious)
                    ioc_type = IOCType.REGISTRY_KEY
                elif placeholder == '{port}':
                    value = str(random.choice(list(SUSPICIOUS_PORTS.keys())))
                    ioc_type = None  # Port numbers aren't IOCs by themselves
                elif placeholder == '{malware}':
                    value = random.choice(MALWARE_FAMILIES)
                    ioc_type = None  # Malware names aren't IOCs by themselves
                else:
                    continue
                
                replacements[placeholder] = value
        
        # Build the final text and track IOC positions
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        # Find positions of IOCs in final text
        for placeholder, value in replacements.items():
            if placeholder in ['{ip}', '{domain}', '{url}', '{hash}', '{email}', 
                              '{cve}', '{filepath}', '{registry}']:
                start_pos = text.find(value)
                if start_pos != -1:
                    end_pos = start_pos + len(value)
                    # Determine IOC type based on placeholder
                    if placeholder == '{ip}':
                        labels.append((start_pos, end_pos, IOCType.IPV4))
                    elif placeholder == '{domain}':
                        labels.append((start_pos, end_pos, IOCType.DOMAIN))
                    # ... etc
        
        risk_score = malicious_ratio
        context_type = 'malicious' if malicious_ratio > 0.5 else 'mixed'
        
        return SyntheticSample(
            text=text,
            labels=labels,
            risk_score=risk_score,
            context_type=context_type
        )
    
    def generate_benign_text(self) -> SyntheticSample:
        """Generate completely benign text."""
        templates = [
            "The system update completed successfully at {time}.",
            "User {username} logged in from workstation {hostname}.",
            "Database backup completed. Size: {size}GB. Duration: {duration} minutes.",
            "Application version {version} installed successfully.",
            "Network latency to server: {latency}ms. Packet loss: {loss}%.",
        ]
        
        template = random.choice(templates)
        text = template.format(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username=random.choice(["admin", "user", "john.doe", "jane.smith"]),
            hostname=f"WORKSTATION{random.randint(1, 100):03d}",
            size=round(random.uniform(1, 100), 2),
            duration=random.randint(1, 60),
            version=f"{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
            latency=random.randint(1, 100),
            loss=round(random.uniform(0, 5), 2)
        )
        
        return SyntheticSample(
            text=text,
            labels=[],
            risk_score=0.0,
            context_type='benign'
        )
    
    def generate_batch(self, n_samples: int = 1000, 
                      malicious_ratio: float = 0.3,
                      benign_ratio: float = 0.3) -> List[SyntheticSample]:
        """Generate batch of synthetic samples.
        
        Args:
            n_samples: Total number of samples
            malicious_ratio: Ratio of purely malicious samples
            benign_ratio: Ratio of purely benign samples
            (remaining will be mixed)
        """
        samples = []
        
        n_malicious = int(n_samples * malicious_ratio)
        n_benign = int(n_samples * benign_ratio)
        n_mixed = n_samples - n_malicious - n_benign
        
        # Generate malicious samples
        for _ in range(n_malicious):
            samples.append(self.generate_context_with_iocs(
                n_iocs=random.randint(1, 5),
                malicious_ratio=1.0
            ))
        
        # Generate benign samples
        for _ in range(n_benign):
            samples.append(self.generate_benign_text())
        
        # Generate mixed samples
        for _ in range(n_mixed):
            samples.append(self.generate_context_with_iocs(
                n_iocs=random.randint(1, 3),
                malicious_ratio=random.uniform(0.3, 0.7)
            ))
        
        random.shuffle(samples)
        return samples
    
    def export_to_jsonl(self, samples: List[SyntheticSample], filepath: str):
        """Export samples to JSONL format for training."""
        with open(filepath, 'w') as f:
            for sample in samples:
                record = {
                    'text': sample.text,
                    'labels': [(s, e, t.value) for s, e, t in sample.labels],
                    'risk_score': sample.risk_score,
                    'context_type': sample.context_type
                }
                f.write(json.dumps(record) + '\n')