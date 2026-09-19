"""Narrow GitHub Project #4 client for Broadway operational metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

PROJECT_ID = "PVT_kwHOAZFnCc4Bhhjq"


class ProjectBoardError(RuntimeError):
    """A GitHub Project operation failed with actionable command output."""


@dataclass(frozen=True)
class ProjectItem:
    item_id: str
    draft_id: str
    title: str
    body: str


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectBoardError(f"GitHub Project response has invalid {label}")
    return value


def _identifier(value: object, prefix: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ProjectBoardError(f"GitHub Project response has invalid {label}")
    return value


def _opaque_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectBoardError(f"GitHub Project response has invalid {label}")
    return value


def _project_items_page(after: str | None) -> dict[str, Any]:
    query = (
        "query($p:ID!,$a:String){node(id:$p){... on ProjectV2{items(first:100,after:$a)"
        "{nodes{id content{... on DraftIssue{id title body}}} pageInfo{hasNextPage endCursor}}}}}"
    )
    data = _run_gh_graphql(query, p=_identifier(PROJECT_ID, "PVT_", "project id"), a=after)
    _object(data.get("node"), "project node")
    return data


def _run_gh_graphql(query: str, **variables: str | None) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    command.extend(f"-f={key}={value}" for key, value in variables.items() if value is not None)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ProjectBoardError(f"GitHub Project command failed ({result.returncode}): {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProjectBoardError(f"GitHub Project returned invalid JSON: {result.stdout.strip()}") from error
    if payload.get("errors"):
        raise ProjectBoardError(f"GitHub Project GraphQL error: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ProjectBoardError("GitHub Project response lacks GraphQL data")
    return data


def iter_items() -> Iterator[ProjectItem]:
    """Yield every editable draft item, including pages beyond GitHub's first 100."""
    after: str | None = None
    while True:
        node = _project_items_page(after).get("node") or {}
        items = node.get("items") or {}
        for item in items.get("nodes") or []:
            content = item.get("content") or {}
            draft_id = content.get("id")
            if isinstance(draft_id, str):
                yield ProjectItem(item["id"], draft_id, str(content.get("title", "")), str(content.get("body", "")))
        page = items.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            return
        after = page.get("endCursor")
        if not isinstance(after, str) or not after:
            raise ProjectBoardError("GitHub Project pagination returned no cursor")


def list_items(status_name: str | None = None) -> list[ProjectItem]:
    """Return all editable draft items, optionally matching their Status field."""
    if status_name is None:
        return list(iter_items())
    query = (
        "query($p:ID!,$a:String){node(id:$p){... on ProjectV2{items(first:100,after:$a)"
        "{nodes{id fieldValueByName(name:\"Status\"){... on ProjectV2ItemFieldSingleSelectValue{name}} "
        "content{... on DraftIssue{id title body}}} pageInfo{hasNextPage endCursor}}}}}"
    )
    after: str | None = None
    matches: list[ProjectItem] = []
    while True:
        data = _run_gh_graphql(query, p=_identifier(PROJECT_ID, "PVT_", "project id"), a=after)
        node = _object(data.get("node"), "project node")
        items = _object(node.get("items"), "project items")
        for item in items.get("nodes") or []:
            content = item.get("content") or {}
            status = item.get("fieldValueByName") or {}
            if status.get("name") == status_name and isinstance(content.get("id"), str):
                matches.append(ProjectItem(item["id"], content["id"], str(content.get("title", "")), str(content.get("body", ""))))
        page = items.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            return matches
        after = page.get("endCursor")
        if not isinstance(after, str) or not after:
            raise ProjectBoardError("GitHub Project pagination returned no cursor")


def get(item_id: str) -> ProjectItem:
    query = "query($i:ID!){node(id:$i){... on ProjectV2Item{id content{... on DraftIssue{id title body}}}}}"
    item = _object(_run_gh_graphql(query, i=item_id).get("node"), "item node")
    if _identifier(item.get("id"), "PVTI_", "item id") != item_id:
        raise ProjectBoardError("GitHub Project response item id does not match request")
    content = _object(item.get("content"), "draft issue")
    draft_id = _opaque_id(content.get("id"), "draft issue id")
    if not isinstance(content.get("title"), str) or not isinstance(content.get("body"), str):
        raise ProjectBoardError("GitHub Project response has invalid draft issue body")
    return ProjectItem(item_id, draft_id, content["title"], content["body"])


def search_items(query: str, *, field: str = "both", exact: bool = False) -> list[ProjectItem]:
    if field not in {"title", "body", "both"}:
        raise ValueError("field must be title, body, or both")
    def matches(value: str) -> bool:
        return value == query if exact else query.casefold() in value.casefold()
    return [item for item in iter_items() if (field in {"title", "both"} and matches(item.title)) or (field in {"body", "both"} and matches(item.body))]


def find_state_mirror(record_id: str) -> str | None:
    matches = []
    for item in iter_items():
        try:
            body = json.loads(item.body)
            if not isinstance(body, dict):
                raise ProjectBoardError(f"GitHub Project mirror body for {item.item_id} is not an object")
            if body.get("id") == record_id:
                matches.append(item.item_id)
        except json.JSONDecodeError:
            continue
    if len(matches) > 1:
        raise ProjectBoardError(f"duplicate GitHub Project mirrors for {record_id}: {matches}")
    return matches[0] if matches else None


def create_draft(title: str, body: str, status_name: str | None = None) -> str:
    mutation = "mutation($p:ID!,$t:String!,$b:String!){addProjectV2DraftIssue(input:{projectId:$p,title:$t,body:$b}){projectItem{id}}}"
    payload = _run_gh_graphql(mutation, p=_identifier(PROJECT_ID, "PVT_", "project id"), t=title, b=body)
    item_id = _identifier(_object(_object(payload.get("addProjectV2DraftIssue"), "create draft").get("projectItem"), "created item").get("id"), "PVTI_", "created item id")
    if status_name is not None:
        update_status(item_id, status_name)
    return item_id


def update_draft(item_id: str, *, title: str | None = None, body: str | None = None) -> None:
    current = get(item_id)
    mutation = "mutation($i:ID!,$t:String!,$b:String!){updateProjectV2DraftIssue(input:{draftIssueId:$i,title:$t,body:$b}){draftIssue{id}}}"
    payload = _run_gh_graphql(mutation, i=current.draft_id, t=title if title is not None else current.title, b=body if body is not None else current.body)
    updated = _object(payload.get("updateProjectV2DraftIssue"), "draft update")
    draft = _object(updated.get("draftIssue"), "updated draft issue")
    if _opaque_id(draft.get("id"), "updated draft issue id") != current.draft_id:
        raise ProjectBoardError("GitHub Project response draft id does not match update")


def _status_option_id(status_name: str) -> tuple[str, str]:
    query = (
        "query($p:ID!,$a:String){node(id:$p){... on ProjectV2{fields(first:100,after:$a)"
        "{nodes{... on ProjectV2SingleSelectField{id name options{id name}}} pageInfo{hasNextPage endCursor}}}}}"
    )
    after: str | None = None
    while True:
        fields = (_run_gh_graphql(query, p=PROJECT_ID, a=after).get("node") or {}).get("fields") or {}
        for field in fields.get("nodes") or []:
            if field.get("name") != "Status":
                continue
            for option in field.get("options") or []:
                if option.get("name") == status_name and isinstance(option.get("id"), str):
                    return field["id"], option["id"]
            raise ProjectBoardError(f"GitHub Project Status has no option named {status_name!r}")
        page = fields.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        after = page.get("endCursor")
        if not isinstance(after, str) or not after:
            raise ProjectBoardError("GitHub Project pagination returned no cursor")
    raise ProjectBoardError("GitHub Project has no single-select Status field")


def update_status(item_id: str, status_name: str) -> None:
    field_id, option_id = _status_option_id(status_name)
    mutation = "mutation($p:ID!,$i:ID!,$f:ID!,$o:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$o}}){projectV2Item{id}}}"
    payload = _run_gh_graphql(mutation, p=PROJECT_ID, i=item_id, f=field_id, o=option_id)
    updated = _object(payload.get("updateProjectV2ItemFieldValue"), "status update")
    item = _object(updated.get("projectV2Item"), "updated project item")
    if _identifier(item.get("id"), "PVTI_", "updated project item id") != item_id:
        raise ProjectBoardError("GitHub Project response item id does not match status update")


def void_duplicate(item_id: str, canonical_item_id: str, reason: str, done_status: str = "Done") -> None:
    if not reason.strip():
        raise ValueError("duplicate reason must be nonempty")
    current = get(item_id)
    body = f"{current.body.rstrip()}\n\nVOID — duplicate of {canonical_item_id}: {reason.strip()}"
    update_draft(item_id, body=body)
    update_status(item_id, done_status)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    get_command = sub.add_parser("get"); get_command.add_argument("item_id")
    list_command = sub.add_parser("list"); list_command.add_argument("--status")
    search = sub.add_parser("search"); search.add_argument("query"); search.add_argument("--field", default="both", choices=("title", "body", "both")); search.add_argument("--exact", action="store_true")
    create = sub.add_parser("create-draft"); create.add_argument("title"); create.add_argument("body"); create.add_argument("--status")
    update = sub.add_parser("update-draft"); update.add_argument("item_id"); update.add_argument("--title"); update.add_argument("--body")
    status = sub.add_parser("update-status"); status.add_argument("item_id"); status.add_argument("status_name")
    void = sub.add_parser("void-duplicate"); void.add_argument("item_id"); void.add_argument("canonical_item_id"); void.add_argument("reason")
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "get": result: object = get(args.item_id).__dict__
        elif args.command == "list": result = [item.__dict__ for item in list_items(args.status)]
        elif args.command == "search": result = [item.__dict__ for item in search_items(args.query, field=args.field, exact=args.exact)]
        elif args.command == "create-draft": result = {"item_id": create_draft(args.title, args.body, args.status)}
        elif args.command == "update-draft": update_draft(args.item_id, title=args.title, body=args.body); result = {"item_id": args.item_id}
        elif args.command == "update-status": update_status(args.item_id, args.status_name); result = {"item_id": args.item_id}
        else: void_duplicate(args.item_id, args.canonical_item_id, args.reason); result = {"item_id": args.item_id}
    except (OSError, ValueError, ProjectBoardError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"PROJECT BOARD ERROR: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
