//! Small roxmltree helpers mirroring ElementTree's `find`/`findall` on
//! direct children. Like ElementTree without a namespace map, a plain tag
//! only matches elements that have no namespace.

use roxmltree::Node;

/// Element children of `n` named `tag`.
pub(crate) fn children<'a, 'i>(n: Node<'a, 'i>, tag: &'static str) -> impl Iterator<Item = Node<'a, 'i>> {
    n.children()
        .filter(move |c| c.is_element() && c.tag_name().namespace().is_none() && c.tag_name().name() == tag)
}

/// First element child of `n` named `tag`.
pub(crate) fn child<'a, 'i>(n: Node<'a, 'i>, tag: &'static str) -> Option<Node<'a, 'i>> {
    children(n, tag).next()
}

/// `n.find("a/b/c")`.
pub(crate) fn find<'a, 'i>(n: Node<'a, 'i>, path: &[&'static str]) -> Option<Node<'a, 'i>> {
    path.iter().try_fold(n, |at, tag| child(at, tag))
}
