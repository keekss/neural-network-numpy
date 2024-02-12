from dependency_injector import containers, providers
from services import Foo
import yaml

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()  # Configuration section name

    foo = providers.Singleton(
        lambda a, b: Foo(a, b)  # Unpack the configuration options
    )

# Load the configuration from the YAML file
with open('config.yml', 'r') as f:
    config_data = yaml.safe_load(f)

container = Container()
container.config.update(config_data)  # Update the configuration

# Get the Foo service and call the bar method
foo_service = container.foo(container.config.Foo.a(), container.config.Foo.b())
foo_service.bar()