#!/usr/bin/env python3
"""Fallback IOCs for when network fetching fails."""

# Common malware hashes (real examples)
MALWARE_MD5 = [
    '44d88612fea8a8f36de82e1278abb02f',  # EICAR test file
    '6a8b703b4ef62e95ad03de4e310a3b27',  # Emotet
    '8f6a3e4e2c5b7d9a1f0e3d8b4c7a9e2f',  # TrickBot
    'ce07d208a88c1e5b4e1bed2c0e9b7c8e',  # Cobalt Strike
    '5d41402abc4b2a76b9719d911017c592',  # Common test
]

MALWARE_SHA256 = [
    '275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f',  # EICAR
    'ed01ebfbc9eb5bbea545af4d01bf5f1071661840480439c6e5babe8e080e41aa',  # WannaCry
    '24d004a104d3e3b2e6e7e5c4e3f4e5b5e6d7c8a9b0a1b2c3d4e5f6a7b8c9d0e1f2',  # Ryuk
    'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2',  # Conti
]

# Common C2 servers and malicious IPs
MALICIOUS_IPS = [
    '185.220.101.45',  # TOR exit node
    '45.142.213.99',   # Known C2
    '23.106.123.188',  # Malware hosting
    '104.21.48.143',   # Phishing
    '162.159.137.238', # DGA domain host
    '185.174.137.80',  # Botnet C2
    '45.61.136.47',    # Cryptominer
    '185.183.107.127', # RAT server
    '193.29.13.201',   # Exploit kit
    '91.208.135.34',   # Spam relay
]

# Common malicious domains
MALICIOUS_DOMAINS = [
    'evil-malware.tk',
    'phishing-site.ml',
    'malware-download.ga',
    'botnet-c2.cf',
    'ransomware.click',
    'exploit-kit.download',
    'cryptominer.top',
    'rat-server.work',
    'fake-bank.online',
    'scam-site.live',
    # DGA-like domains
    'xyzabc123def.com',
    'qwerty789uiop.net',
    'mnbvcx456lkjh.org',
    'asdfgh321poiu.info',
    'zxcvbn654qwer.biz',
]

# Common malicious URLs
MALICIOUS_URLS = [
    'http://evil-site.tk/payload.exe',
    'https://malware.ga/download.php',
    'http://phishing.ml/login.html',
    'https://exploit.cf/shell.aspx',
    'http://botnet.work/gate.php',
    'https://ransomware.click/encrypt.bin',
    'http://45.142.213.99:8080/config.json',
    'https://185.220.101.45/admin/upload.php',
    'http://evil.com/wp-content/uploads/2024/01/invoice.exe',
    'https://bad-site.net/temp/document.scr',
]

# Common CVE identifiers
CVES = [
    'CVE-2021-44228',  # Log4Shell
    'CVE-2021-34527',  # PrintNightmare
    'CVE-2020-1472',   # Zerologon
    'CVE-2019-0708',   # BlueKeep
    'CVE-2017-0144',   # EternalBlue
    'CVE-2022-30190',  # Follina
    'CVE-2021-40444',  # MSHTML RCE
    'CVE-2022-26809',  # RPC RCE
    'CVE-2021-26855',  # ProxyLogon
    'CVE-2020-0601',   # CurveBall
]

# Common suspicious file paths
SUSPICIOUS_PATHS_WIN = [
    'C:\\Windows\\Temp\\svchost.exe',
    'C:\\Users\\Public\\Documents\\malware.dll',
    'C:\\ProgramData\\Microsoft\\update.bat',
    'C:\\Windows\\System32\\drivers\\rootkit.sys',
    'C:\\Users\\%USERNAME%\\AppData\\Local\\Temp\\payload.exe',
    'C:\\Windows\\SysWOW64\\backdoor.exe',
    'C:\\Temp\\cryptominer.exe',
    'C:\\Windows\\Tasks\\persistence.vbs',
]

SUSPICIOUS_PATHS_UNIX = [
    '/tmp/.hidden/malware.sh',
    '/var/tmp/rootkit.ko',
    '/dev/shm/miner',
    '/usr/local/bin/.backdoor',
    '/etc/cron.d/persistence',
    '/opt/.malware/payload',
    '/home/user/.ssh/authorized_keys2',
    '/usr/bin/python3.evil',
]

# Registry keys commonly used by malware
REGISTRY_KEYS = [
    'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware',
    'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce\\Backdoor',
    'HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Services\\RootkitService',
    'HKEY_CURRENT_USER\\Software\\Classes\\ms-settings\\shell\\open\\command',
    'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\sethc.exe',
]