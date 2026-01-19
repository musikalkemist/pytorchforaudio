# Project Contributors

Thank you to everyone who has contributed to the success of this project! Your efforts, whether large or small, are greatly appreciated.

---

## Community & Project Maintainers

* **musikalkemist:** Author and project creator.
* **HimanshuKGP007:** Project reviewer.
* **MLsound:**  Project maintainer.
    - **Refactoring & Maintenance:**
        - Updated `README.md` to include the full course structure and navigation links.
        - Added `CONTRIBUTING.md` and `CONTRIBUTORS.md` guidelines.
        - Aligned logic between project folders and refined `.gitignore`.
        - Updated dependencies for Python 3.11 compatibility.
        - Unified device selection logic across scripts and centralized dataset configuration.
        - Solved CUDA deserialization errors for CPU-only devices.
    - **Version Management:** Established and organized the legacy branch to preserve the original course environment for students, ensuring compatibility with legacy video content while moving the main repository to modern standards.
    - **Features:**
        - Created an automated dataset downloader for UrbanSound8K.
        - Ensured `StrPath` compatibility in audio data loading.

### Contributors

* **yearat:**  Implemented `.to(device)` support (PR #2, #3, #4, #5).
* **jvaleroliet:** Removed redundant SoftMax function when using CrossEntropyLoss (PR #10).
* **pgq18:** Reported issues related to SoftMax usage (Issue #9).
* **luisriera:** Fixed CUDA device mismatch error in `_resample_if_necessary` (Issue #11).
* **jkarenko:** Contributions to device compatibility (PR #6).

---

Want to see your name on this list? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file to learn how you can help!
