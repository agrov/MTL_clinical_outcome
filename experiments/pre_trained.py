def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    r"""
    Instantiate a pretrained pytorch model from a pre-trained model configuration.
    The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
    train the model, you should first set it back in training mode with ``model.train()``.
    The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
    pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    task.
    The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
    weights are discarded.
    Parameters:
        pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
            Can be either:
                - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                  a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                  this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                  as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                  a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                  arguments ``config`` and ``state_dict``).
        model_args (sequence of positional arguments, `optional`):
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
        config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
            Can be either:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.
            Configuration for the model to use instead of an automatically loaded configuation. Configuration can
            be automatically loaded when:
                - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                  model).
                - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                  by supplying the save directory.
                - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                  configuration JSON file named `config.json` is found in the directory.
        state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
            A state dictionary to use instead of a state dictionary loaded from saved weights file.
            This option can be used if you want to create a model from a pretrained configuration but load your own
            weights. In this case though, you should check if using
            :func:`~transformers.PreTrainedModel.save_pretrained` and
            :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
        cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Load the model weights from a TensorFlow checkpoint save file (see docstring of
            ``pretrained_model_name_or_path`` argument).
        force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received files. Will attempt to resume the download if such a
            file exists.
        proxies (:obj:`Dict[str, str], `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
        local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        mirror(:obj:`str`, `optional`, defaults to :obj:`None`):
            Mirror source to accelerate downloads in China. If you are from China and have an accessibility
            problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
            Please refer to the mirror site for more information.
        kwargs (remaining dictionary of keyword arguments, `optional`):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
            automatically loaded:
                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                  underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                  already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                  initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                  with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                  attribute will be passed to the underlying model's ``__init__`` function.
    .. note::
        Passing :obj:`use_auth_token=True` is required when you want to use a private model.
    .. note::
        Activate the special `"offline-mode"
        <https://huggingface.co/transformers/installation.html#offline-mode>`__ to use this method in a firewalled
        environment.
    Examples::
        >>> from transformers import BertConfig, BertModel
        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained('./test/saved_model/')
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
        >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
    """
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    mirror = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)

    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            *model_args,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                # Load from a TF 1.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                # Load from a TF 2.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    f"Error no file named {[WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + '.index']} found in "
                    f"directory {pretrained_model_name_or_path} or `from_tf` set to False."
                )
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif os.path.isfile(pretrained_model_name_or_path + ".index"):
            if not from_tf:
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                    "from_tf to True to load from this checkpoint."
                )
            archive_file = pretrained_model_name_or_path + ".index"
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                revision=revision,
                mirror=mirror,
            )

        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )
        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
            )
            raise EnvironmentError(msg)

        if resolved_archive_file == archive_file:
            logger.info(f"loading weights file {archive_file}")
        else:
            logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
    else:
        resolved_archive_file = None

    config.name_or_path = pretrained_model_name_or_path

    # Instantiate model.

    if is_deepspeed_zero3_enabled():
        import deepspeed

        logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
        # this immediately partitions the model to avoid the overhead in time and memory copying it on CPU or each GPU first
        with deepspeed.zero.Init():
            model = cls(config, *model_args, **model_kwargs)
    else:
        model = cls(config, *model_args, **model_kwargs)

    if state_dict is None and not from_tf:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
        if resolved_archive_file.endswith(".index"):
            # Load from a TensorFlow 1.X checkpoint - provided by original authors
            model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
        else:
            # Load from our TensorFlow 2.0 checkpoints
            try:
                from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
            except ImportError:
                logger.error(
                    "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                    "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                )
                raise
    else:
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]
            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "error_msgs": error_msgs,
        }
        return model, loading_info

    return model
