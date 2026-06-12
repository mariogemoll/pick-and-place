// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface LicenseRule {
	/** File patterns (glob-like) or specific filenames */
	files: string[];
	/** SPDX license identifier */
	license: string;
	/** Optional: more specific description */
	description?: string;
}

export interface LicenseConfig {
	/** Directories to exclude from processing */
	excludedDirs: string[];
	/** Specific files to exclude */
	excludedFiles: string[];
	/** Copyright text for SPDX-FileCopyrightText header */
	copyrightText: string;
	/** License rules, processed in order (first match wins) */
	rules: LicenseRule[];
}

export const config: LicenseConfig = {
  excludedDirs: ['node_modules', 'dist', 'build', '.git', '.claude', '.venv', '.ipynb_checkpoints', '.pytest_cache', 'SO-ARM100'],
  excludedFiles: ['.DS_Store'],
  copyrightText: '2026 Mario Gemoll',
  rules: [
    // All files with 0BSD
    {
      files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx', '**/*.mjs', '**/*.py', '**/*.sh', '**/*.html', '**/*.css', '**/*.md', '**/*.ipynb'],
      license: '0BSD',
      description: 'All project files'
    }
  ]
};
