# frozen_string_literal: true

Gem::Specification.new do |s|
  s.name          = "PyFed"
  s.version       = "0.0.28"
  s.license       = "MIT"
  s.authors       = ["Amirreza Sokhankhosh"]
  s.email         = ["amirreza.sokhankhosh@gmail.com"]
  s.homepage      = "https://amirrezasokhankhosh.github.io/PyFed/"
  s.summary       = "PyFed is an open-source framework for federated learning"

  s.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^((_includes|_layouts|_sass|assets)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end

  s.required_ruby_version = ">= 2.4.0"

  s.platform = Gem::Platform::RUBY
  s.add_runtime_dependency "jekyll", "> 3.5", "< 5.0"
  s.add_runtime_dependency "jekyll-seo-tag", "~> 2.0"
  s.add_development_dependency "html-proofer", "~> 3.0"
  s.add_development_dependency "rubocop-github", "~> 0.16"
  s.add_development_dependency "w3c_validators", "~> 1.3"
end
