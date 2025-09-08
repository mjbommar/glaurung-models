# External Data Source Ideas (from v1 and Web Research)

This document lists the external datasets, APIs, and data sources that were identified in the `archive/v1` of this project, supplemented with web research. This can serve as a starting point for creating a comprehensive dataset for training models to detect risky or dangerous strings.

## APIs and Live Feeds (Threat Intelligence)

These feeds provide real-time Indicators of Compromise (IOCs) like malicious IPs, domains, URLs, and file hashes.

- **URLhaus (abuse.ch):** Fetches recent and online malicious URLs.
- **Feodo Tracker (abuse.ch):** Fetches a blocklist of botnet Command & Control (C2) IPs.
- **SSL Blacklist (abuse.ch):** Fetches a blacklist of malicious SSL certificate IPs.
- **ThreatFox (abuse.ch):** Fetches recent IOCs associated with malware.
- **The Spamhaus Project:** Real-time blocklists of malicious IPs and domains.
- **FireHOL IP Lists:** Aggregates over 400 publicly available IP feeds.
- **OpenPhish:** Provides a feed of verified phishing URLs.
- **AlienVault OTX (Open Threat Exchange):** Community-powered threat intelligence platform.
- **MISP (Malware Information Sharing Platform):** An open-source platform for sharing IOCs.

## External Datasets (Hosted on GitHub and other sites)

These are static or periodically updated datasets containing IOCs and malware information.

- **FireEye Sunburst Countermeasures:** A list of malicious hashes related to the Sunburst backdoor.
- **Botvrij.eu:** A raw list of domain-based IOCs.
- **Neo23x0/signature-base:** A collection of hash-based IOCs.
- **awesome-iocs (GitHub):** A curated list of IOC resources.
- **MalwareBazaar (abuse.ch):** A project for sharing malware samples and their hashes.
- **VirusShare:** A large repository of malware samples and their corresponding hashes.

## Malicious Network Traffic Datasets (PCAPs)

These datasets are crucial for analyzing payloads from HTTP requests, packets, etc.

- **University of Portsmouth Dataset:** Includes 913 malicious network traffic PCAPs.
- **Kaggle Network Traffic Data for Malicious Activity Detection:** Contains labeled packets, distinguishing normal from flood attacks.
- **Kaggle Malware Detection in Network Traffic Data:** Labeled network flows from the Stratosphere Laboratory, identifying malware-related connections.
- **Mendeley Composed Encrypted Malicious Traffic Dataset:** A balanced collection of encrypted malicious and legitimate traffic.
- **Canadian Institute for Cybersecurity (CIC) Datasets:**
    - **CIC-IDS-2017:** Contains benign traffic and a wide range of common attacks.
    - **CIC-AndMal-2017:** Focuses on network traffic from Android malware.

## Malware and Benign Binary Datasets

These datasets are ideal for extracting strings from binaries to train detection models.

- **MaleX:** A large, curated dataset with 1,044,394 Windows executable binaries (864,669 malware and 179,725 benign).
- **DikeDataset:** Labeled benign and malicious Portable Executable (PE) and Object Linking and Embedding (OLE) files, including malware family labels.
- **Kaggle Malware Detection Dataset:** Contains 50,000 malware and 50,000 benign PE files.
- **Awesome-Malware-Benign-Datasets (GitHub):** A curated list of various malware and benign datasets, including image-based and binary samples.

## Local System Data Sources

For generating "clean" (non-malicious) training data, the system can use local files:

- **System Dictionary:** The file `/usr/share/dict/words` can be used to generate natural-sounding text that is free of IOCs.