# Add Image Input

This guide shows how to include images in user messages using `ImageContent`.

## Overview

Models that support image input (those with `"image"` in `model.input_types`) can process `ImageContent` blocks alongside text. Images are sent as base64-encoded data with a MIME type.

## Creating an ImageContent block

```python
from pi_llm import ImageContent

image = ImageContent(
    data="<base64-encoded image data>",
    mime_type="image/jpeg",
)
```

## Including images in a UserMessage

```python
from pi_llm import UserMessage, TextContent, ImageContent

message = UserMessage(
    content=[
        TextContent(text="Describe this image:"),
        ImageContent(data=base64_data, mime_type="image/png"),
    ],
    timestamp=int(time.time() * 1000),
)
```

## Full example

Content coming soon.

## Next steps

- [Messages & Context](../concepts/messages-and-context.md) -- Content block types
- [Streaming](../concepts/streaming.md) -- Making LLM calls with image context
