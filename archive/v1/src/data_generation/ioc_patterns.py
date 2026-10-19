"""IOC pattern definitions and validation functions."""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IOCType(Enum):
    """Enumeration of IOC types."""
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    DOMAIN = "domain"
    URL = "url"
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    EMAIL = "email"
    CVE = "cve"
    FILE_PATH_WIN = "file_path_win"
    FILE_PATH_UNIX = "file_path_unix"
    REGISTRY_KEY = "registry_key"
    BITCOIN_ADDRESS = "bitcoin_address"
    YARA_RULE = "yara_rule"
    MUTEX = "mutex"
    USER_AGENT = "user_agent"
    BENIGN = "benign"


@dataclass
class IOCPattern:
    """IOC pattern definition."""
    type: IOCType
    pattern: str
    validator: Optional[callable] = None
    risk_weight: float = 1.0


# Core IOC regex patterns
IOC_PATTERNS = {
    IOCType.IPV4: IOCPattern(
        type=IOCType.IPV4,
        pattern=r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        risk_weight=0.7
    ),
    IOCType.IPV6: IOCPattern(
        type=IOCType.IPV6,
        pattern=r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
        risk_weight=0.7
    ),
    IOCType.DOMAIN: IOCPattern(
        type=IOCType.DOMAIN,
        pattern=r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b',
        risk_weight=0.8
    ),
    IOCType.URL: IOCPattern(
        type=IOCType.URL,
        pattern=r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        risk_weight=0.9
    ),
    IOCType.MD5: IOCPattern(
        type=IOCType.MD5,
        pattern=r'\b[a-fA-F0-9]{32}\b',
        risk_weight=0.95
    ),
    IOCType.SHA1: IOCPattern(
        type=IOCType.SHA1,
        pattern=r'\b[a-fA-F0-9]{40}\b',
        risk_weight=0.95
    ),
    IOCType.SHA256: IOCPattern(
        type=IOCType.SHA256,
        pattern=r'\b[a-fA-F0-9]{64}\b',
        risk_weight=0.95
    ),
    IOCType.SHA512: IOCPattern(
        type=IOCType.SHA512,
        pattern=r'\b[a-fA-F0-9]{128}\b',
        risk_weight=0.95
    ),
    IOCType.EMAIL: IOCPattern(
        type=IOCType.EMAIL,
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        risk_weight=0.6
    ),
    IOCType.CVE: IOCPattern(
        type=IOCType.CVE,
        pattern=r'CVE-\d{4}-\d{4,}',
        risk_weight=0.85
    ),
    IOCType.FILE_PATH_WIN: IOCPattern(
        type=IOCType.FILE_PATH_WIN,
        pattern=r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
        risk_weight=0.7
    ),
    IOCType.FILE_PATH_UNIX: IOCPattern(
        type=IOCType.FILE_PATH_UNIX,
        pattern=r'(?:/[^/\0]+)+/?',
        risk_weight=0.7
    ),
    IOCType.REGISTRY_KEY: IOCPattern(
        type=IOCType.REGISTRY_KEY,
        pattern=r'HK(?:EY_)?(?:LOCAL_MACHINE|CURRENT_USER|CLASSES_ROOT|USERS|CURRENT_CONFIG)(?:\\[^\\]+)+',
        risk_weight=0.8
    ),
    IOCType.BITCOIN_ADDRESS: IOCPattern(
        type=IOCType.BITCOIN_ADDRESS,
        pattern=r'\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b',
        risk_weight=0.9
    ),
    IOCType.USER_AGENT: IOCPattern(
        type=IOCType.USER_AGENT,
        pattern=r'Mozilla/[0-9]\.[0-9] \([^)]+\) [^\s]+(?:\s+[^\s]+)*',
        risk_weight=0.5
    ),
}


def validate_ipv4(ip: str) -> bool:
    """Validate IPv4 address."""
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        try:
            num = int(part)
            if num < 0 or num > 255:
                return False
        except ValueError:
            return False
    return True


def validate_domain(domain: str) -> bool:
    """Validate domain name."""
    if len(domain) > 253:
        return False
    labels = domain.split('.')
    if len(labels) < 2:
        return False
    for label in labels:
        if not label or len(label) > 63:
            return False
        if label.startswith('-') or label.endswith('-'):
            return False
    return True


def validate_hash(hash_str: str, expected_length: int) -> bool:
    """Validate cryptographic hash."""
    if len(hash_str) != expected_length:
        return False
    try:
        int(hash_str, 16)
        return True
    except ValueError:
        return False


def extract_iocs(text: str) -> List[Tuple[IOCType, str, int, int]]:
    """Extract all IOCs from text.
    
    Returns:
        List of tuples (ioc_type, matched_text, start_pos, end_pos)
    """
    results = []
    
    for ioc_type, pattern_def in IOC_PATTERNS.items():
        pattern = re.compile(pattern_def.pattern, re.IGNORECASE)
        for match in pattern.finditer(text):
            results.append((
                ioc_type,
                match.group(),
                match.start(),
                match.end()
            ))
    
    return results


def calculate_risk_score(iocs: List[Tuple[IOCType, str, int, int]]) -> float:
    """Calculate overall risk score based on IOCs found."""
    if not iocs:
        return 0.0
    
    total_weight = 0.0
    for ioc_type, _, _, _ in iocs:
        if ioc_type in IOC_PATTERNS:
            total_weight += IOC_PATTERNS[ioc_type].risk_weight
    
    # Normalize to 0-1 range
    max_possible = len(iocs) * 1.0
    return min(total_weight / max_possible, 1.0) if max_possible > 0 else 0.0


# Common benign patterns that might be confused with IOCs
BENIGN_PATTERNS = [
    # Common version numbers that might look like IPs
    r'\b\d{1,2}\.\d{1,2}\.\d{1,2}\b',
    # Product codes that might look like hashes
    r'\b[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}\b',
    # Common file extensions
    r'\.(txt|pdf|doc|docx|xls|xlsx|png|jpg|jpeg|gif|mp4|avi|zip|rar)\b',
]


# Suspicious TLDs often used in malicious campaigns
SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.click', '.download',
    '.top', '.work', '.date', '.racing', '.win', '.bid'
}


# Common malware families for synthetic data generation
MALWARE_FAMILIES = [
    'Emotet', 'TrickBot', 'Ryuk', 'Conti', 'REvil', 'Darkside',
    'Cobalt Strike', 'Mimikatz', 'BloodHound', 'PowerShell Empire',
    'Metasploit', 'Lazarus', 'APT28', 'APT29', 'Carbanak'
]


# Common ports associated with malicious activity
SUSPICIOUS_PORTS = {
    22: 'SSH',
    23: 'Telnet',
    445: 'SMB',
    1433: 'MSSQL',
    3389: 'RDP',
    4444: 'Metasploit',
    5985: 'WinRM',
    8080: 'HTTP Proxy',
    9001: 'Tor',
}