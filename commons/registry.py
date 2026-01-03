class Registry:
    mapping = {
        "handler_mapping": {},
    }

    @classmethod
    def register_handler(cls, name):
        def wrap(builder_cls):
            from handlers.base import BaseHandler

            assert issubclass(
                builder_cls, BaseHandler
            ), "All chatbots must inherit BaseChatBot class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["handler_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["handler_mapping"][name]
                    )
                )
            cls.mapping["handler_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def get_handler_class(cls, name):
        return cls.mapping["handler_mapping"].get(name, None)

    @classmethod
    def list_handlers(cls):
        return sorted(cls.mapping["handler_mapping"].keys())


registry = Registry()
