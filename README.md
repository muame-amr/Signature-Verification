<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--****
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/amrnumenor/signature-verification">
    <img src="./icon.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Signature Verification</h3>

  <p align="center">
    Smart offline signature validation using deep learning
    <br />
    <a href="https://github.com/amrnumenor/signature-verification"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/amrnumenor/signature-verification">View Demo</a>
    ·
    <a href="https://github.com/amrnumenor/signature-verification/issues">Report Bug</a>
    ·
    <a href="https://github.com/amrnumenor/signature-verification/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
</details>

<p align="right">(<a href="#top">back to top</a>)</p>

## About the Project

 <img src="results/gui.gif"  width="600" height="480">

Signature verification is a process flow of verifying unique signatures automatically and instantly to determine whether the signature is a valid one or forged. Offline or static verification is the process of validating a document signature after it has been made. The signature in question will be compared to the one that stored in database. Handwritten signature is one of the most generally accepted personal attributes with identity verification for banking or business purpose.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

- [Java](https://www.java.com/en/) - Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.
- [Deeplearning4j](https://deeplearning4j.org/) - Eclipse Deeplearning4j is a programming library written in Java for the Java virtual machine. It is a framework with wide support for deep learning algorithms.
- [IntelliJ IDEA](https://www.jetbrains.com/idea/) - IntelliJ IDEA is an integrated development environment written in Java for developing computer software
- [Apache Maven](https://maven.apache.org/) - Maven is a build automation tool used primarily for Java projects.
- [JavaFX](https://openjfx.io/) - JavaFX is a software platform for creating and delivering desktop applications, as well as rich web applications that can run across a wide variety of devices.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

### Prerequisites

1. Install Java

   Download Java JDK [here](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).

   (Note: Use Java 8 for full support of DL4J operations)

   Check the version of Java using:

   ```sh
   java -version
   ```

2. Install IntelliJ IDEA Community Edition

   Download and install [IntelliJ IDEA](https://www.jetbrains.com/idea/download/).

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/amrnumenor/signature-verification
   ```
2. Open project in IntelliJ IDEA
3. Wait until the process of resolving dependencies done
4. Setting JDK 1.8 in IntelliJ IDEA

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

1. Run `src/main/java/application/GUI.java` in IntelliJ IDEA
2. Choose sign image file from your PC
3. Click verify

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- Collect [Data](https://www.kaggle.com/robinreni/signature-verification-dataset) - Retrieved from kaggle datasets
- Data labeling - Two classes: **Valid** & **Forged**
- Image transformation - Horizontal flip, Image rotation, Scaling
- Modeling - CNN, Transfer Leaning(VGG16)
  - Model Configuration
  - Hyperparameter Tuning
- Inference using test data
- Build GUI

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

- Looi Yao Wei - looiyaowei@gmail.com
- Yong Xian Pang - xian-0712@hotmail.com
- Muhammad Amiruddin - m.amiruddin27@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## References

- Alajrami, E., Ashqar, B. A., Abu-Nasser, B. S., Khalil, A. J., Musleh, M. M., Barhoom, A. M., & Abu-Naser, S. S. (2020). Handwritten signature verification using deep learning. International Journal of Academic Multidisciplinary Research (IJAMR), 3(12).
- Poddar, J., Parikh, V., & Bharti, S. K. (2020). Offline signature recognition and forgery detection using deep learning. Procedia Computer Science, 170, 610-617.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/amrnumenor/signature-verification.svg?style=for-the-badge
[contributors-url]: https://github.com/amrnumenor/signature-verification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/amrnumenor/signature-verification.svg?style=for-the-badge
[forks-url]: https://github.com/amrnumenor/signature-verification/network/members
[stars-shield]: https://img.shields.io/github/stars/amrnumenor/signature-verification.svg?style=for-the-badge
[stars-url]: https://github.com/amrnumenor/signature-verification/stargazers
[issues-shield]: https://img.shields.io/github/issues/amrnumenor/signature-verification.svg?style=for-the-badge
[issues-url]: https://github.com/amrnumenor/signature-verification/issues
[license-shield]: https://img.shields.io/github/license/amrnumenor/signature-verification.svg?style=for-the-badge
[license-url]: https://github.com/amrnumenor/signature-verification/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
