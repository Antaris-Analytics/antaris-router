# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Antaris Router, please report it responsibly:

**Email**: security@antarisanalytics.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and provide updates on remediation timeline.

## Security Considerations

### Data Storage
- All routing data is stored locally in JSON files
- No network calls made during routing decisions  
- Cost tracking data contains usage patterns but no prompt content
- Configuration files may contain model API costs (public information)

### Dependencies
- Zero external dependencies reduces attack surface
- Uses only Python standard library modules
- No network dependencies or external service calls

### Input Validation
- Prompt classification uses text analysis only
- No code execution of user input
- JSON configuration files are parsed safely
- File paths are validated to prevent directory traversal

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |

## Best Practices

When using Antaris Router:

1. **File Permissions**: Secure routing data files with appropriate permissions
2. **Configuration**: Review model definitions before adding custom models  
3. **Monitoring**: Regularly review cost tracking data for unexpected patterns
4. **Updates**: Keep the package updated to receive security patches